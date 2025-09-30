import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from kitti_util import *
from run_nsg_helper import get_rays, get_rays_np, box_pts
from collections import defaultdict
import imageio.v2 as imageio
from tqdm import tqdm

class kitti_tracking_dataset(torch.utils.data.Dataset):
    def __init__(self, img2_root, selected_frames, device, near=0.5, far=1e10,
                 box_scale=1.0, progressive_param=None, data_sample_param=None,
                 use_collider=True, debug_cam=False, kitti2vkitti=True, load_image=True,
                 N_samples_plane=0, plane_type=None, N_rand=8192, 
                 bound_setting=0, expand_bound_factor=1):
        super().__init__()

        self.img2_root = img2_root
        self.selected_frames = selected_frames
        self.init_id_relationship()
        self.device = device
        self.near = near
        self.far = far
        self.box_scale = box_scale
        self.use_collider = use_collider
        self.use_plane_sample = N_samples_plane > 0 and plane_type is not None
        self.load_image = load_image
        self.N_rand = N_rand
        self.bound_setting = bound_setting
        self.expand_bound_factor = expand_bound_factor

        sequence = img2_root[-4:]
        data_root = img2_root[:-13]
        self.img3_root = os.path.join(os.path.join(data_root, 'image_03'), sequence)
        calib_path = os.path.join(os.path.join(data_root, 'calib'), sequence+'.txt')
        oxts_path = os.path.join(os.path.join(data_root, 'oxts'), sequence+'.txt')
        label_path = os.path.join(os.path.join(data_root, 'label_02'), sequence+'.txt')
        preprocess_path = os.path.join(os.path.join(data_root, 'autorf'), sequence)
        dense_depth_path02 = os.path.join(os.path.join(data_root, 'depth_02_npy'), sequence)
        dense_depth_path03 = os.path.join(os.path.join(data_root, 'depth_03_npy'), sequence)

        self.calib = Calibration(calib_path=calib_path)
        self.poses_imu_w = self.get_imu_poses_calibration(oxts_path)
        cam_poses = self.get_camera_poses(debug_cam=debug_cam)
        self.visible_objects, self.objects_meta = self.get_obj_poses(label_path)

        if kitti2vkitti:
            # Align axis with vkitti axis
            KITTI2VKITTI = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])  # rot pi_2 about x axis
            self.poses = KITTI2VKITTI @ cam_poses
            self.visible_objects[:, :, [9]] *= -1
            self.visible_objects[:, :, [7, 8, 9]] = self.visible_objects[:, :, [7, 9, 8]]
        else:
            self.poses = cam_poses
        self.half = len(self.poses) // 2

        if self.load_image:
            self.images = self.get_images()
            self.H, self.W = self.images.shape[1], self.images.shape[2]
        else:
            self.imagefilenames = self.get_imagefilenames()
            test_load_image = imageio.imread(self.imagefilenames[0])
            self.H, self.W = test_load_image.shape[0], test_load_image.shape[1]

        if self.use_plane_sample:
            self.use_collider = False

        # Important Notice: self.split is only computed in left images.
        # Do not forget to add right images in later process.
        # Namely, function self.get_bounds_from_depth and self.scale_bounds.
        if progressive_param is not None:
            self.split = self.progressive_split(progressive_param)
        else:
            self.split = [0]
        
        print('scaling...')
        self.min_bounds, self.max_bounds = [], []
        self.origins, self.pose_scale_factors, self.scene_bounds = [], [], []
        self.rays_colliders = []
        self.plane_bds_dicts = []
        for i in range(len(self.split)):
            s = self.split[i]
            e = self.split[i+1] if i != len(self.split)-1 else self.half
            min_bound, max_bound = self.get_bounds_from_depth(preprocess_path, s, e)
            origin, pose_scale_factor, scene_bounds = self.scale_bounds(min_bound, max_bound, s, e)
            pose_scale_factor *= self.expand_bound_factor
            scene_bounds *= self.expand_bound_factor
            if self.use_collider:
                self.rays_colliders.append(NearFarCollider(near/pose_scale_factor, far/pose_scale_factor, scene_bounds, self.device))
            elif self.use_plane_sample:
                poses = np.concatenate([self.poses[s:e], self.poses[s+self.half:e+self.half]], axis=0)
                plane_bds, plane_normal, plane_delta, id_planes, near, far = plane_bounds(
                    poses, plane_type, near/pose_scale_factor, far/pose_scale_factor, N_samples_plane)
                plane_bds_dict = {
                    'plane_bds': torch.tensor(plane_bds, dtype=torch.float32, device=device),
                    'plane_normal': torch.tensor(plane_normal, dtype=torch.float32, device=device),
                    'plane_delta': torch.tensor(plane_delta, dtype=torch.float32, device=device),
                    'id_planes': torch.tensor(id_planes, dtype=torch.float32, device=device),
                    'near': near,
                    'far': far
                }
                self.plane_bds_dicts.append(plane_bds_dict)
            self.min_bounds.append(min_bound)
            self.max_bounds.append(max_bound)
            self.origins.append(origin)
            self.pose_scale_factors.append(pose_scale_factor)
            self.scene_bounds.append(scene_bounds)
        self.min_bounds = np.concatenate(self.min_bounds)
        self.max_bounds = np.concatenate(self.max_bounds)

        print('getting mask, depth, masked patch...')
        self.masks = self.get_masks(preprocess_path)
        self.depths = self.get_depth(preprocess_path)
        # self.depths = self.get_dense_depth(dense_depth_path02, dense_depth_path03)
        self.img_transform = T.Compose([T.Resize((80, 120)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.patches, self.patch_masks = self.get_patch_mask(preprocess_path)
        print('remove objs with no good 2D mask')
        self.remove_obj()

        if data_sample_param is None:
            data_sample_param = {
                'train_every': 1,
                'test_every': -1,
                'is_train': True,
                'render_test': True,
                'side': 'both'
            }
        self.data_sample(data_sample_param)
        self.is_train = data_sample_param['is_train']
        self.render_test = data_sample_param['render_test']

        if self.is_train:
            print('creating train indices and shuffling...')
            self.train_indices = np.indices((len(self.i_train), self.H, self.W)).reshape((3, -1)).T
            np.random.shuffle(self.train_indices)


    def is_valid_frame_id(self, frame_id):
        ''' input: absolute id
        '''
        frame_num = len(os.listdir(self.img2_root))
        for s, e in zip(self.selected_frames[0], self.selected_frames[1]):
            if s <= frame_id <= e or s <= frame_id-frame_num <= e:
                return True
        return False
    
    def init_id_relationship(self):
        # to_relative: absolute frame id to relative frame id
        # from_relative: relative frame id to absolute frame id
        frame_num = len(os.listdir(self.img2_root))
        start, end = self.selected_frames
        self.to_ralative, self.from_relative = {}, []
        relative_id = 0
        for seq_i, seq_u in enumerate(start):
            for i in range(seq_u, end[seq_i]+1):
                self.to_ralative[i] = relative_id
                self.from_relative.append(i)
                relative_id += 1
        for seq_i, seq_u in enumerate(start):
            for i in range(seq_u, end[seq_i]+1):
                self.to_ralative[i+frame_num] = relative_id
                self.from_relative.append(i+frame_num)
                relative_id += 1

    def debug_plot(self):

        # plt.figure()
        # plt.scatter(self.poses_imu_w[:, 0, 3], self.poses_imu_w[:, 1, 3], color='orange')
        # plt.scatter(self.poses_velo_w[:, 0, 3], self.poses_velo_w[:, 1, 3], color='black')
        # plt.show()

        half = self.poses.shape[0] // 2
        
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.scatter(self.scaled_poses[:half, 0, 3], self.scaled_poses[:half, 1, 3], self.scaled_poses[:half, 2, 3], color='green')
        # ax.scatter(self.scaled_poses[half:, 0, 3], self.scaled_poses[half:, 1, 3], self.scaled_poses[half:, 2, 3], color='red')
        # plt.show()

        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        axs[0].scatter(self.scaled_poses[:half, 0, 3], self.scaled_poses[:half, 2, 3], color='green')
        axs[0].set_title('scaled poses')
        axs[0].set_aspect('equal')
        axs[1].scatter(self.poses[:half, 0, 3], self.poses[:half, 2, 3], color='red')
        axs[1].set_title('poses')
        axs[1].set_aspect('equal')
        fig0 = plt.figure()
        ax = fig0.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(self.scaled_poses[:half, 0, 3], self.scaled_poses[:half, 1, 3], self.scaled_poses[:half, 2, 3], color='green')
        ax.set_title('3d scaled poses')
        
        # figs = []
        # for i in range(len(self.split)):
        #     figs.append(plt.figure())
        #     axi = figs[i].add_subplot(111, projection='3d')
        #     axi.set_xlabel('X')
        #     axi.set_ylabel('Y')
        #     axi.set_ylabel('Z')
        #     s = self.split[i]
        #     e = self.split[i+1] if i != len(self.split)-1 else self.half
        #     axi.scatter(self.poses[s:e, 0, 3], self.poses[s:e, 1, 3], self.poses[s:e, 2, 3], color='red')
        #     axi.set_title('split_'+str(i))
        #     obj_mask = np.where(np.bitwise_and(s<=self.visible_objects[:, :, 0], self.visible_objects[:, :, 0]<e))
        #     objs = self.visible_objects[obj_mask[0], obj_mask[1]]
        #     axi.scatter(objs[:, 7], objs[:, 8], objs[:, 9], color='orange')
        #     axi.scatter(self.scene_bounds[i][:, 0], self.scene_bounds[i][:, 1], self.scene_bounds[i][:, 2], color='green')
        #     axi.set_aspect('equal')
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.scatter(self.poses[:half, 0, 3], self.poses[:half, 1, 3], self.poses[:half, 2, 3], color='red')
        ax1.set_title('3d poses')
        
        obj_mask = np.where(self.visible_objects[:, :, 0]!=-1)
        objs = self.visible_objects[obj_mask[0], obj_mask[1]]

        ax1.scatter(objs[:, 7], objs[:, 8], objs[:, 9], color='orange')
        # ax1.scatter(self.min_bounds[0, 0], self.min_bounds[0, 1], self.min_bounds[0, 2], color='green')
        # ax1.scatter(self.max_bounds[0, 0], self.max_bounds[0, 1], self.max_bounds[0, 2], color='green')

        trans = self.poses[0, :3, 3]
        rot = self.poses[0, :3, :3]
        for i, color in zip(range(3), ['blue', 'green', 'black']):
            ax1.quiver(trans[0], trans[1], trans[2], rot[0, i], rot[1, i], rot[2, i], color=color)
        
        ax1.set_aspect('equal')
        plt.show()

    def get_imu_poses_calibration(self, oxts_path)->list:
        def latlon_to_mercator(lat, lon, s):
            r = 6378137.0
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return x, y
        
        oxt_data = np.loadtxt(oxts_path)
        poses_imu = []

        scale = None
        for frame_id, oxt in enumerate(oxt_data):
            if not self.is_valid_frame_id(frame_id): continue
            if scale is None:
                lat0 = oxt[0]
                scale = np.cos(lat0 * np.pi / 180)
            pose_imu_i = np.eye(4)
            x, y = latlon_to_mercator(oxt[0], oxt[1], scale)
            z = oxt[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxt[3], oxt[4], oxt[5])
            pose_imu_i[:3, :] = np.hstack((rotation, translation[:, None]))
            poses_imu.append(pose_imu_i)

        return poses_imu

    def get_camera_poses(self, debug_cam=False)->np.ndarray:
        imu2velo = np.vstack((self.calib.I2V, [0, 0, 0, 1]))
        velo2cam_base = np.vstack((self.calib.V2C, [0, 0, 0, 1]))
        cam_base2camrect = np.eye(4)
        cam_base2camrect[:3, :3] = self.calib.R0
        camdebug = np.eye(4)
        if debug_cam:
            sequence_number = int(self.img2_root[-4:])
            # Debug Camera Offset
            if sequence_number == 2:
                yaw = np.deg2rad(0.7) ## Affects camera rig roll: High --> counterclockwise
                pitch = np.deg2rad(-0.5) ## Affects camera rig yaw: High --> Turn Right
                # pitch = np.deg2rad(-0.97)
                roll = np.deg2rad(0.9) ## Affects camera rig pitch: High -->  up
                # roll = np.deg2rad(1.2)
            elif sequence_number == 1:
                yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
                pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
                roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            else:
                yaw = np.deg2rad(0.05)
                # pitch = np.deg2rad(-0.75)
                pitch = np.deg2rad(-0.97)
                roll = np.deg2rad(1.05)
                #roll = np.deg2rad(1.2)
            camdebug[:3, :3] = get_rotation(roll, pitch, yaw)
        P22imu = np.linalg.inv(self.calib.camrect2cam2 @ camdebug @ cam_base2camrect @ velo2cam_base @ imu2velo)
        P32imu = np.linalg.inv(self.calib.camrect2cam3 @ camdebug @ cam_base2camrect @ velo2cam_base @ imu2velo)
        OPENCV_TO_OPENGL = np.array([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]])  # kitti use OPENGL axis
        cam_poses = []
        for pose_imu_i in self.poses_imu_w:
            # pose_imu_i: imu2world
            cam_poses.append(pose_imu_i @ P22imu @ OPENCV_TO_OPENGL)
            # --> P22world
        for pose_imu_i in self.poses_imu_w:
            cam_poses.append(pose_imu_i @ P32imu @ OPENCV_TO_OPENGL)
        
        return np.array(cam_poses)

    def get_obj_poses(self, label_path):
        '''
        Return:
            visible_objects
                frame_id, cam_id, obj_id, obj_type, dim, xyz, yaw
            objects_meta: dict, keys are different obj_ids
                values:
                obj_id, obj_l, obj_h, obj_w, obj_type
        '''
        _sem2label = {
            'Misc': -1,
            'Car': 0,
            'Van': 0,
            'Truck': 2,
            'Tram': 3,
            'Pedestrian': 4
        }
        camera_ls = [2, 3]
        with open(label_path, 'r') as read_label:
            label_lines = read_label.readlines()
        # with 'DontCare'
        objs_all_raw = [Object3d(line) for line in label_lines]
        # without 'DontCare', with relative img_id in order to match self.poses_imu_w
        objs_all = defaultdict(list)
        objects_meta = {}
        n_obj = np.zeros(len(self.poses_imu_w))

        for obj in objs_all_raw:
            if obj.obj_id < 0 or not self.is_valid_frame_id(obj.img_id):
                continue
            if obj.type in _sem2label:
                objs_all[self.to_ralative[obj.img_id]].append(obj)
                n_obj[self.to_ralative[obj.img_id]] += 1

        max_n_obj = int(n_obj.max())
        visible_objects = np.ones([len(self.poses_imu_w)*2, max_n_obj, 11]) * -1
        
        for relative_img_id, objs in objs_all.items():
            for obj in objs:
                obj_type = _sem2label[obj.type]
                if obj_type != 4:
                    obj_dim = [obj.dim[0] * self.box_scale, obj.dim[1], obj.dim[2] * self.box_scale]
                else:
                    obj_dim = [obj.dim[0] * 1.2, obj.dim[1], obj.dim[2] * 1.2]
                if obj.obj_id not in objects_meta:
                    objects_meta[obj.obj_id] = np.array([obj.obj_id] + obj_dim + [obj_type])
            
                obj_pose_cam = np.eye(4)
                obj_pose_cam[:3, :3] = roty(obj.ry)
                obj_pose_cam[:3, 3] = np.array(obj.t)
                # obj2cam_base

                obj_pose_imu = np.vstack((self.calib.V2I, [0, 0, 0, 1])) @ \
                               np.vstack((self.calib.C2V, [0, 0, 0, 1])) @ \
                               obj_pose_cam
                
                pose_obj_w_i = self.poses_imu_w[relative_img_id] @ obj_pose_imu

                yaw_approx = - np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])

                for j, cam in enumerate(camera_ls):
                    obj_array = np.array([obj.img_id + j*len(os.listdir(self.img2_root)), cam, obj.obj_id, obj_type] + \
                                         obj_dim + pose_obj_w_i[:3, 3].tolist() + [yaw_approx])
                    tmp_id0 = relative_img_id + j*len(self.poses_imu_w)
                    tmp_id1 = np.argwhere(visible_objects[tmp_id0, :, 0] < 0).min()
                    visible_objects[tmp_id0, tmp_id1] = obj_array

        print('Removing not moving objects')
        obj_to_del = []
        for key, values in objects_meta.items():
            all_obj_poses = np.where(visible_objects[:, :, 2] == key)
            if len(all_obj_poses[0]) > 0 and values[4] != 4.:
                x = all_obj_poses[0][[0, -1]]
                y = all_obj_poses[1][[0, -1]]
                obj_poses_0 = visible_objects[x[0], y[0]][7:10]
                obj_poses_1 = visible_objects[x[1], y[1]][7:10]
                distance = np.linalg.norm(obj_poses_1 - obj_poses_0)
                print(f'obj {key} moved {distance} meters in selected frames')
                if distance < 0.5:
                    print('Removed obj:', key)
                    obj_to_del.append(key)
                    visible_objects[all_obj_poses] = np.ones(11) * -1.
        for key in obj_to_del:
            del objects_meta[key]
        
        return visible_objects, objects_meta

    def get_bounds_from_depth(self, depth_root, s, e):
        ''' compute on both cam2 and cam3
        '''
        cur_min_bounds = None
        cur_max_bounds = None
        for filename in sorted(os.listdir(depth_root)):
            frame_id = int(filename.split('_')[0])
            if not self.is_valid_frame_id(frame_id): continue
            i = self.to_ralative[frame_id]
            obj_id = int(filename.split('_')[1])
            if (s <= i < e or s <= i-self.half < e) and obj_id == -2:
                _, rays_d = get_rays_np(self.H, self.W, self.calib.P2[:3, :3], self.poses[i])
                rays_d = np.reshape(rays_d, [-1, 3])
                rays_o = self.poses[i, :3, 3][None, ...]
                depth = np.load(os.path.join(depth_root, filename))
                depth = np.reshape(depth, [-1, 1])

                depth_mask = np.squeeze(depth > 0)
                filtered_directions = rays_d[depth_mask]
                filtered_depth = depth[depth_mask]

                points = rays_o + filtered_directions * filtered_depth
                if cur_min_bounds is not None:
                    bounds = np.concatenate((points, rays_o, cur_min_bounds, cur_max_bounds))
                else:
                    bounds = np.concatenate((points, rays_o))
                
                cur_min_bounds = bounds.min(axis=0).reshape(1, 3)
                cur_max_bounds = bounds.max(axis=0).reshape(1, 3)
        return cur_min_bounds, cur_max_bounds

    def scale_bounds(self, min_bound, max_bound, s, e):
        origin = (max_bound + min_bound) * 0.5
        if self.bound_setting == 0:
            pose_scale_factor = np.linalg.norm((max_bound - min_bound) * 0.5)
        else:
            pose_scale_factor = (max_bound - min_bound) * 0.5
            pose_scale_factor = pose_scale_factor.reshape(3)
        for i in range(len(self.poses)):
            if s <= i < e or s <= i-self.half < e:
                self.poses[i, :3, 3] -= origin.flatten()
                self.poses[i, :3, 3] /= pose_scale_factor
                self.visible_objects[i, :, 7:10] -= origin
                self.visible_objects[i, :, 7:10] /= pose_scale_factor
                self.visible_objects[i, :, 4:7] /= pose_scale_factor
        scene_bounds = (np.concatenate([min_bound, max_bound]) - origin) / pose_scale_factor
        if self.bound_setting != 0:
            pose_scale_factor = np.linalg.norm(pose_scale_factor)
        return origin, pose_scale_factor, scene_bounds

    def get_masks(self, mask_root):
        res = []
        for filename in sorted(os.listdir(mask_root)):
            if not filename.endswith('.png'): continue
            frame_id = int(filename.split('_')[0])
            obj_id  = int(filename.split('_')[1])
            if self.is_valid_frame_id(frame_id) and obj_id == -1:
                res.append(imageio.imread(os.path.join(mask_root, filename))[:, :, 0])
        return np.array(res)
    
    def get_depth(self, depth_root):
        res = []
        for filename in sorted(os.listdir(depth_root)):
            if not filename.endswith('.npy'): continue
            frame_id = int(filename.split('_')[0])
            obj_id  = int(filename.split('_')[1])
            if self.is_valid_frame_id(frame_id) and obj_id == -2:
                res.append(np.load(os.path.join(depth_root, filename)))
        return np.array(res)
    
    def get_dense_depth(self, left_root, right_root):
        res = []
        valid_filenames = []
        for filename in sorted(os.listdir(left_root)):
            if not filename.endswith('.npy'): continue
            frame_id = int(filename.split('.')[0])
            if self.is_valid_frame_id(frame_id):
                valid_filenames.append(os.path.join(left_root, filename))
        for filename in sorted(os.listdir(right_root)):
            if not filename.endswith('.npy'): continue
            frame_id = int(filename.split('.')[0])
            if self.is_valid_frame_id(frame_id):
                valid_filenames.append(os.path.join(right_root, filename))
        for f in tqdm(valid_filenames):
            res.append(np.load(f))
        return np.array(res)

    
    def get_patch_mask(self, mask_root):
        res = {}
        res_mask = {}
        tmp_mask = None
        for filename in sorted(os.listdir(mask_root)):
            if not filename.endswith('.png'): continue
            frame_id = int(filename.split('_')[0])
            obj_id  = int(filename.split('_')[1])
            img_type = filename.split('_')[-1]
            if self.is_valid_frame_id(frame_id) and obj_id in self.objects_meta.keys():
                if img_type == 'mask.png':
                    tmp_mask = imageio.imread(os.path.join(mask_root, filename)).astype(bool)
                elif img_type == 'patch.png':
                    tmp_patch = imageio.imread(os.path.join(mask_root, filename))
                    tmp_patch[~tmp_mask] = 0
                    obj_type = self.objects_meta[obj_id][-1]
                    hw_scale = tmp_patch.shape[0] / tmp_patch.shape[1]
                    if obj_type != 4 and hw_scale > 5.0:
                        print(f'skip frame {frame_id}, obj {obj_id} when extracting patches')
                        continue
                    balanced_patch = self.img_transform(T.ToTensor()(tmp_patch)).to(self.device)
                    res[(frame_id, obj_id)] = balanced_patch
                    res_mask[(frame_id, obj_id)] = tmp_mask
        return res, res_mask

    def remove_obj(self):
        for i in range(self.visible_objects.shape[0]):
            for j in range(self.visible_objects.shape[1]):
                tmp = self.visible_objects[i, j].copy()
                if tmp[0] == -1: continue
                if (tmp[0], tmp[2]) in self.patches:
                    continue
                self.visible_objects[i, j] = np.ones_like(tmp) * -1
                if tmp[1] == 2:
                    print(f'removed cam2, frame {int(tmp[0])}, obj {int(tmp[2])}.')
                elif tmp[1] == 3:
                    print(f'removed cam3, frame {int(tmp[0]-len(os.listdir(self.img2_root)))}({int(tmp[0])}), obj {int(tmp[2])}.')
                

    def get_images(self):
        imgs = []
        for frame_root in [self.img2_root, self.img3_root]:
            for frame_id, filename in enumerate(sorted(os.listdir(frame_root))):
                if self.is_valid_frame_id(frame_id):
                    imgs.append(imageio.imread(os.path.join(frame_root, filename)))
        imgs = np.array(imgs).clip(0, 255)
        imgs = (imgs / 255.).astype(np.float32)

        return imgs

    def get_imagefilenames(self):
        filenames = []
        for frame_root in [self.img2_root, self.img3_root]:
            for frame_id, filename in enumerate(sorted(os.listdir(frame_root))):
                if self.is_valid_frame_id(frame_id):
                    filenames.append(os.path.join(frame_root, filename))

        return filenames


    def progressive_split(self, progressive_param):
        '''
        Split the whole scene to several small scene.
        focus on poses[:, [0, 2], 3]
        for scaled_poses, -z <==> forward, -x <==> left, y <==> dontcare
        Important! Till now, need manually adjust split results
        '''
        
        pose_inv = invert_transformation(self.poses[0])
        self.scaled_poses = []
        for pose_i in self.poses:
            self.scaled_poses.append(pose_inv@pose_i)
        self.scaled_poses = np.array(self.scaled_poses)
        t = progressive_param['t']
        angle = progressive_param['angle']
        print('progressive params are t={0:02f}m, rot={1:02f} degree'.format(t, angle))
        relative_poses = np.stack([pose_inv@pose_i for pose_i in self.poses[:self.half]])
        origins = [relative_poses[0]]
        origin_frame_ids = [0]
        origin_index = 0
        i = 0
        while i < self.half:
            if np.abs(relative_poses[i][2, 3]-origins[origin_index][2, 3]) > t:
                origin_index += 1
                origin_frame_ids.append(i)
                origins.append(relative_poses[i])
                pose_inv = invert_transformation(self.poses[i])
                relative_poses[i:] = np.array([pose_inv@pose_i for pose_i in self.poses[i:self.half]])
                print('split by t', end='\t')
            elif np.abs(np.arctan2(relative_poses[i][2, 0], relative_poses[i][2, 2])) > np.deg2rad(angle):
                origin_index += 1
                origin_frame_ids.append(i)
                origins.append(relative_poses[i])
                pose_inv = invert_transformation(self.poses[i])
                relative_poses[i:] = np.array([pose_inv@pose_i for pose_i in self.poses[i:self.half]])
                print('split by rot', end='\t')
            i += 1
        num_in_part = []
        for i in range(len(origin_frame_ids)-1):
            num_in_part.append(origin_frame_ids[i+1] - origin_frame_ids[i])
        num_in_part.append(self.half - origin_frame_ids[-1])
        print('number of frames in each small scene', num_in_part)

        return origin_frame_ids
    
    def data_sample(self, data_sample_param):
        frame_num = self.half
        side = data_sample_param['side']
        train_every = data_sample_param['train_every']
        test_every = data_sample_param['test_every']
        assert train_every * test_every < 0
        if train_every > 0:
            i_train_cam2 = np.arange(0, frame_num, train_every)
            i_train_cam3 = np.arange(frame_num, frame_num*2, train_every)
            i_test_cam2 = np.setdiff1d(np.arange(0, frame_num), i_train_cam2, assume_unique=True)
            i_test_cam3 = np.setdiff1d(np.arange(frame_num, frame_num*2), i_train_cam3, assume_unique=True)
        else:
            i_test_cam2 = np.arange(0, frame_num, test_every)
            i_test_cam3 = np.arange(frame_num, frame_num*2, test_every)
            i_train_cam2 = np.setdiff1d(np.arange(0, frame_num), i_test_cam2, assume_unique=True)
            i_train_cam3 = np.setdiff1d(np.arange(frame_num, frame_num*2), i_test_cam3, assume_unique=True)
        if side == 'both':
            self.i_train = np.hstack([i_train_cam2, i_train_cam3])
            self.i_test = np.hstack([i_test_cam2, i_test_cam3])
        elif side == 'left':
            self.i_train = i_train_cam2
            self.i_test = i_test_cam2
        elif side == 'right':
            self.i_train = i_train_cam3
            self.i_test = i_test_cam3
        else:
            raise NotImplementedError
        print('side:', side)
        print('i_train:', self.i_train)
        print('i_test:', self.i_test)

    def shuffle(self):
        # only shuffle when training
        assert self.is_train
        np.random.shuffle(self.train_indices)

    def __len__(self):
        if self.is_train:
            return len(self.train_indices) // self.N_rand + 1
        elif self.render_test:
            return len(self.i_test)
        else:
            return len(self.i_train)

    def __getitem__(self, idx):
        if self.is_train:
            selected_indices = self.train_indices[idx*self.N_rand:(idx+1)*self.N_rand]
            img_idx, col_idx, row_idx = np.split(selected_indices, 3, axis=-1)
            img_idx = self.i_train[img_idx]
            unq_img, unq_img_idx = np.unique(img_idx, return_inverse=True)
            all_rays_rgb_obj = []
            all_mask = []
            all_depth = []
            all_patch = {}
            all_patch_mask = {}
            all_split_id = []
            for i in range(len(unq_img)):
                current_img_id = unq_img[i]
                tmp_idx = current_img_id if current_img_id < self.half else current_img_id - self.half
                for j in range(len(self.split)):
                    if tmp_idx >= self.split[j]:
                        split_id = j
                pose = torch.tensor(self.poses[current_img_id], dtype=torch.float32, device=self.device)
                rays = get_rays(self.H, self.W, self.calib.P2[:3, :3], pose)
                cols = col_idx[unq_img_idx==i]
                rows = row_idx[unq_img_idx==i]
                rays_o = rays[0][cols, rows, :].reshape(-1, 3)
                rays_d = rays[1][cols, rows, :].reshape(-1, 3)
                if self.load_image:
                    rgb = torch.tensor(self.images[current_img_id, cols, rows], dtype=torch.float32, device=self.device).reshape(-1, 3)
                else:
                    img = imageio.imread(self.imagefilenames[current_img_id])
                    rgb = torch.tensor(img[cols, rows]/255., dtype=torch.float32, device=self.device).reshape(-1, 3)
                    rgb = rgb.clamp(0, 1)
                objs = torch.tensor(self.visible_objects[current_img_id], dtype=torch.float32, device=self.device)
                objs = torch.repeat_interleave(objs.reshape(1, -1), rays_o.shape[0], dim=0)

                if self.use_collider:
                    near, far = self.rays_colliders[split_id](rays_o, rays_d)
                elif self.use_plane_sample:
                    near = self.plane_bds_dicts[split_id]['near'] * torch.ones_like(rays_o[:, :1])
                    far = self.plane_bds_dicts[split_id]['far']* torch.ones_like(rays_o[:, :1])
                else:
                    near = self.near * torch.ones_like(rays_o[:, :1]) / self.pose_scale_factors[split_id]
                    far = self.far * torch.ones_like(rays_o[:, :1]) / self.pose_scale_factors[split_id]

                rays_rgb_obj = torch.cat((rays_o, rays_d, rgb, near, far, objs), dim=-1)

                mask = torch.tensor(self.masks[current_img_id, cols, rows], dtype=torch.float32, device=self.device)
                depth = torch.tensor(self.depths[current_img_id, cols, rows], dtype=torch.float32, device=self.device)
                patch = {}
                patch_mask = {}
                for k in self.patches.keys():
                    if k[0] == self.from_relative[current_img_id]:
                        patch[k] = self.patches[k]
                        patch_mask[k] = self.patch_masks[k]
                
                all_rays_rgb_obj.append(rays_rgb_obj)
                all_mask.append(mask)
                all_depth.append(depth)
                all_patch.update(patch)
                all_patch_mask.update(patch_mask)
                all_split_id.append(split_id * torch.ones(len(rays_o), dtype=torch.int32, device=self.device))
            all_rays_rgb_obj = torch.cat(all_rays_rgb_obj)
            all_mask = torch.cat(all_mask).squeeze()
            all_depth = torch.cat(all_depth).squeeze()
            all_split_id = torch.cat(all_split_id)

            return all_rays_rgb_obj, all_split_id, all_mask, all_depth, all_patch, all_patch_mask

        else:
            if self.render_test:
                i_render = self.i_test
            else:
                i_render = self.i_train
            current_img_id = i_render[idx]
            pose = torch.tensor(self.poses[current_img_id], dtype=torch.float32, device=self.device)
            rays = get_rays(self.H, self.W, self.calib.P2[:3, :3], pose)
            rays_o = rays[0].reshape(-1, 3)
            rays_d = rays[1].reshape(-1, 3)
            if self.load_image:
                rgb = torch.tensor(self.images[current_img_id], dtype=torch.float32, device=self.device).reshape(-1, 3)
            else:    
                img = imageio.imread(self.imagefilenames[current_img_id])
                rgb = torch.tensor(img/255., dtype=torch.float32, device=self.device).reshape(-1, 3)
                rgb = rgb.clamp(0, 1)
            objs = torch.tensor(self.visible_objects[current_img_id], dtype=torch.float32, device=self.device)
            
            tmp_idx = current_img_id if current_img_id < self.half else current_img_id - self.half
            for i in range(len(self.split)):
                if tmp_idx >= self.split[i]:
                    split_id = i
            
            if self.use_collider:
                near, far = self.rays_colliders[split_id](rays_o, rays_d)
            elif self.use_plane_sample:
                near = self.plane_bds_dicts[split_id]['near'] * torch.ones_like(rays_o[:, :1])
                far = self.plane_bds_dicts[split_id]['far']* torch.ones_like(rays_o[:, :1])
            else:
                near = self.near * torch.ones_like(rays_o[:, :1]) / self.pose_scale_factors[split_id]
                far = self.far * torch.ones_like(rays_o[:, :1]) / self.pose_scale_factors[split_id]

            rays_rgb = torch.cat((rays_o, rays_d, rgb, near, far), dim=-1)

            mask = torch.tensor(self.masks[current_img_id], dtype=torch.float32, device=self.device)
            depth = torch.tensor(self.depths[current_img_id], dtype=torch.float32, device=self.device)
            patch = {}
            patch_mask = {}
            for k in self.patches.keys():
                if k[0] == self.from_relative[current_img_id]:
                    patch[k] = self.patches[k]
                    patch_mask[k] = self.patch_masks[k]
            
            return rays_rgb, objs, split_id, mask, depth, patch, pose, patch_mask

class NearFarCollider(nn.Module):
    def __init__(self, near, far, scene_bounds, device):
        super().__init__()
        self.device = device
        self.near = torch.tensor(near, dtype=torch.float32, device=self.device)
        self.far = torch.tensor(far, dtype=torch.float32, device=self.device)
        self.scene_bounds = torch.tensor(scene_bounds, dtype=torch.float32, device=self.device)  # 2 x 3
        self.sphere_center = (self.scene_bounds[1] + self.scene_bounds[0]) * 0.5
        self.sphere_radius = (self.scene_bounds[1] - self.scene_bounds[0]) * np.sqrt(3) / 2
    
    def forward(self, rays_o, rays_d):
        nears = self.near.expand([len(rays_o), 1])
        fars = self.far.expand([len(rays_o), 1])
        
        nears = self.truncate_with_plane_intersection(rays_o, rays_d, self.scene_bounds[1, 2], nears)  # max_bound z
        fars = self.truncate_with_plane_intersection(rays_o, rays_d, self.scene_bounds[0, 2], fars)  # min_bound z

        rays_o_sphere, rays_d_sphere = self.ellipse_to_sphere_coords(rays_o, rays_d)
        _, sphere_fars = self.intersect_with_sphere(rays_o_sphere, rays_d_sphere, torch.zeros(3, device=self.device))
        fars = torch.minimum(fars, sphere_fars)
        
        nears = nears.clamp_min(self.near)
        fars = fars.clamp_min(nears + 1e-6).clamp_max(self.far)

        return nears.to(torch.float32), fars.to(torch.float32)

    def truncate_with_plane_intersection(self, rays_o, rays_d, altitude, default_bounds):
        starts_before = rays_o[..., 2] > altitude
        goes_down = rays_d[..., 2] < 0

        # rays that both start above the altitude and go downward, 
        # indicating possible intersection with the plane
        boundable_rays = torch.minimum(starts_before, goes_down)
        new_bounds = default_bounds.clone()

        ray_points = rays_o[boundable_rays]
        ray_dirs = rays_d[boundable_rays]
        if ray_points.shape[0] == 0:
            # there are no rays that satisfy the conditions for intersection
            return new_bounds

        new_bounds[boundable_rays] = ((altitude - ray_points[..., 2]) / ray_dirs[..., 2]).unsqueeze(-1)

        return new_bounds
    
    def ellipse_to_sphere_coords(self, rays_o, rays_d):
        rays_o_sphere = (rays_o - self.sphere_center) / self.sphere_radius
        rays_d_sphere = rays_d / self.sphere_radius
        return rays_o_sphere, rays_d_sphere

    def intersect_with_sphere(self, rays_o, rays_d, center, radius=1.0, near_plane=0.0):
        oc = rays_o - center
        a = torch.sum(rays_d * rays_d, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(oc * rays_d, dim=-1, keepdim=True)
        c = torch.sum(oc * oc, dim=-1, keepdim=True) - radius * radius

        discriminant = b * b - 4 * a * c
        mask = discriminant > 0

        discriminant = discriminant.masked_fill(~mask, 0)
        sqrt_discriminant = torch.sqrt(discriminant)

        nears = (-b - sqrt_discriminant) / (2.0 * a)
        fars = (-b + sqrt_discriminant) / (2.0 * a)

        nears = nears.masked_fill(~mask, near_plane)
        fars = fars.masked_fill(~mask, near_plane)

        return nears, fars

def get_all_ray_3dbox_intersection(rays, obj, obj2remove=-1):
    
    rays_to_remove = None
    
    mask = box_pts(
        rays=[rays[:, :3], rays[:, 3:6]],
        pose=obj[:, :, 7:10],
        theta_y=obj[:, :, 10],
        dim=obj[:, :, 4:7],
        one_intersec_per_ray=False,
        mask_only=True
    )

    rays_on_obj = torch.any(mask, dim=1).bool()

    return rays_on_obj, rays_to_remove

def resample_rays(rays, rays_bckg, obj, gt, rays_on_obj, obj_mask_idx, scene_objects, objects_meta):

    obj_bckg = obj[~rays_on_obj]
    gt_bckg = {k: v[~rays_on_obj] for k, v in gt.items()}

    if len(rays) == 0:
        return rays_bckg, obj_bckg, gt_bckg, obj_mask_idx
    
    obj = obj[rays_on_obj]
    gt = {k: v[rays_on_obj] for k, v in gt.items()}

    device = rays.device

    mask = box_pts(
        rays=[rays[:, :3], rays[:, 3:6]],
        pose=obj[:, :, 7:10],
        theta_y=obj[:, :, 10],
        dim=obj[:, :, 4:7],
        one_intersec_per_ray=True,
        mask_only=True
    )

    hit_id = obj[:, :, 2].unsqueeze(-1)[mask]

    obj_counts = np.zeros(np.array(scene_objects).max().astype(np.int32) + 2, dtype=np.int32)
    new_rays = {}
    new_obj = {}
    new_gt = {}
    new_obj_mask = {}

    for k in scene_objects+[-1]:
        k_mask = torch.where(hit_id==torch.tensor(k, device=device), \
                             mask, torch.zeros_like(mask))
        k_mask_dim0 = torch.any(k_mask, dim=1).bool()
        k_rays = rays[k_mask_dim0]

        obj_counts[int(k)] += int(k_rays.shape[0])
        new_rays[int(k)] = k_rays
        new_obj[int(k)] = obj[k_mask_dim0]
        new_gt[int(k)] = {k: v[k_mask_dim0] for k, v in gt.items()}
        new_obj_mask[int(k)] = obj_mask_idx[k_mask_dim0]
    
    if len(new_rays[-1]) == len(rays):
        return rays_bckg, obj_bckg, gt_bckg, obj_mask_idx

    unique_classes = torch.unique(obj[:, :, 3])
    class_multiplier = {int(x.cpu()):0 for x in unique_classes if x != -1}
    for x in objects_meta.values():
        obj_id = int(x[0])
        class_id = int(x[-1])
        if class_id in class_multiplier.keys():
            class_multiplier[class_id] += obj_counts[obj_id]
    
    hits_per_class = np.array(list(class_multiplier.values()))

    for key in class_multiplier:
        class_multiplier[key] = np.round((class_multiplier[key] / hits_per_class.max()) ** (-1))

    ret_rays = []
    ret_obj = []
    ret_gt = defaultdict(list)
    ret_obj_mask = []

    for k in scene_objects:
        _id_hit = int(k)
        if obj_counts[_id_hit] > 0:
            _hit_factor = (np.max(obj_counts) // obj_counts[_id_hit]).astype(np.int32)

            # Manually add support for objects not present enough in specific datasets e.g. pedestrians in KITTI sequences
            if objects_meta[_id_hit][4] == 2 or objects_meta[_id_hit][4] == 1:
                _support_factor = class_multiplier[2]
                _hit_factor *= _support_factor
            if objects_meta[_id_hit][4] == 4:
                _support_factor = class_multiplier[4]
                _hit_factor *= _support_factor
            if objects_meta[_id_hit][4] == 0:
                _support_factor = class_multiplier[0]
                _hit_factor *= _support_factor
            
            _hit_factor = np.minimum(_hit_factor, 1e1)
            ret_rays.append(new_rays[_id_hit].repeat_interleave(int(_hit_factor), dim=0))
            ret_obj.append(new_obj[_id_hit].repeat_interleave(int(_hit_factor), dim=0))
            for key in gt.keys():
                ret_gt[key].append(new_gt[_id_hit][key].repeat_interleave(int(_hit_factor), dim=0))
            ret_obj_mask.append(new_obj_mask[_id_hit].repeat_interleave(int(_hit_factor), dim=0))

    ret_rays = torch.cat(ret_rays)
    ret_obj = torch.cat(ret_obj)
    ret_gt = {k: torch.cat(v) for k, v in ret_gt.items()}
    ret_obj_mask = torch.cat(ret_obj_mask)

    if rays_bckg is not None:
        ret_rays = torch.cat((ret_rays, rays_bckg))
        ret_obj = torch.cat((ret_obj, obj_bckg))
        ret_gt = {k: torch.cat((v, gt_bckg[k])) for k, v in ret_gt.items()}
    
    return ret_rays, ret_obj, ret_gt, ret_obj_mask


def plane_bounds(poses, plane_type, near, far, N_samples):
    ''' Define Plane bounds and plane index array

    Args:
        poses: camera poses
        plane_type: selects the specific distribution of samples
        near: closest sampling point along a ray
        far: minimum distance to last plane in the scene
        N_samples: amount of steps along each ray

    Returns:
        plane_bds: first and last sampling plane in the scene
        plane_normal: plane normals
        plane_delta: distance between each plane
        id_planes: id of planes selected for sampling give a specific plane_type
        near: distance to the closest samping point along a ray
        far: distance to last plane in the scene
    '''

    # [N_poses*N_samples, xyz]
    plane_normal = -poses[0, :3, 2]

    # The first plane in front of the first pose in the scene [N_poses, N_samples, xyz]
    first_plane = poses[0, :3, -1] + near * plane_normal

    # Current Assumption the vehicle is driving a straight line
    # For 2 cameras each half of the poses are similar
    n_left = int(poses.shape[0] / 2)

    # Distance between the first and last pose
    max_pose_dist = np.linalg.norm(poses[-1, :3, -1] - poses[0, :3, -1])

    # Distances between two frames
    if not n_left > 1:
        pose_dist = max_pose_dist + 1e-9
    else:
        pose_dist = np.linalg.norm(poses[0:n_left - 1, :3, -1] - poses[1:n_left, :3, -1], axis=1)

    if plane_type == 'uniform':
        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    if plane_type == 'uniform_exp':
        # The first plane in front of the first pose in the scene [N_poses, N_samples, xyz]
        first_plane = poses[0, :3, -1] + near * 1.2 * plane_normal

        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    elif plane_type == 'experimental':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 1.7
            id_planes[i] = np.sum(t[:i + 1])

        id_planes[-1] = np.sum(t)
        plane_delta = (far - near) / id_planes[-1]
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'double':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 2
            id_planes[i] = np.sum(t[:i + 1])

        id_planes[-1] = np.sum(t)
        plane_delta = (far - near) / id_planes[-1]
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'bckg' or plane_type == 'reversed':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 1.
            id_planes[i] = np.ceil(np.sum(t[:i + 1]))

        id_planes[-1] = np.ceil(np.sum(t))
        id_planes = np.sort((id_planes[-1]-id_planes))
        plane_delta = (far - near) / id_planes[-1]

        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'non-uniform':
        # Adds depth+1*delta between each plane
        t = np.linspace(0, N_samples - 1, N_samples)
        id_planes = np.zeros([N_samples])
        for i in range(N_samples):
            id_planes[i] = np.sum(t[:i + 1])

        plane_delta = (far - near) / (id_planes[-1])
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'strict_uniform':
        first_plane = poses[-1, :3, -1] + near * plane_normal
        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    elif plane_type == 'move':
        aprox_near_planes = round(max_pose_dist / near)
        aprox_delta = (far-near) / (N_samples-1)
        no_near_spaces = int(max_pose_dist / aprox_delta)+1
        if no_near_spaces > 1.:
            print('Selected planes might not work')

        plane_delta = near
        planes_per_section = np.ceil(aprox_delta / near)
        id_planes = np.linspace(0, N_samples - 1, N_samples) * planes_per_section

        add_planes = (no_near_spaces-1) * planes_per_section

    elif plane_type == 'static_move':
        first_plane = poses[n_left-1, :3, -1] + near * 1.2 * plane_normal

        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)
        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    last_plane = first_plane + ((id_planes[-1] + add_planes) * plane_delta) * plane_normal
    far = near + plane_delta * (id_planes[-1] + add_planes)
    plane_bds = np.concatenate([first_plane[:, None], last_plane[:, None]], axis=1)

    return plane_bds, plane_normal, plane_delta, id_planes, near, far


if __name__ == '__main__':
    seq = '6'
    local_path = 'E:\\BaiduNetdiskDownload\\kitti_tracking_test\\image_02\\000' + seq
    local_path = '/data0/dataset/kitti_tracking/training/image_02/0006'
    progressive_param = {
        't': 30.0,
        'angle': 30.0,
        'min_num': 10
    }
    data_sample_param = {
        'train_every': 1,
        'test_every': -1,
        'is_train': True,
        'render_test': False
    }
    dataset = kitti_tracking_dataset(
        local_path, [[0], [20]], device='cuda:1',
        progressive_param=progressive_param,
        data_sample_param=data_sample_param,
        box_scale=1.9)
    print('initialized dataset')


    # rays, obj, split_id, mask, depth, patch, pose = dataset.__getitem__(264)
    # H, W = dataset.H, dataset.W
    # near_np = rays[..., 9].reshape(H, W).cpu().numpy()
    # far_np = rays[..., 10].reshape(H, W).cpu().numpy()
    # depth_np = depth.reshape(H, W).cpu().numpy() / dataset.pose_scale_factors[split_id]

    # np.savetxt('near.txt', near_np)
    # np.savetxt('far.txt', far_np)
    # np.savetxt('depth.txt', depth_np)

    # train_sampler = data_sampler(dataset, 2, -1, True)
    # train_batch_sampler = torch.utils.data.BatchSampler(
    #     train_sampler,
    #     batch_size=8,
    #     drop_last=False
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_sampler = train_batch_sampler,
    #     collate_fn = collate_lambda
    # )

    # N_rand = 1024
    # N_iters = 500000

    # last_t = time.time()
    # i_batch = 0
    # random_indices = torch.randperm(dataset[0][0].shape[0])
    
    # for i in range(0, N_iters):
    #     for rays, objs, split_id, mask, depth_gt, patch in train_dataloader:
    #         # rays.shape [bs, H*W, 11]:
    #         #   rays_o*3, rays_d*3, rgb*3, near, far
    #         # obj.shape [bs, max_obj, 11]:
    #         #   frame_id, cam_id, obj_id, obj_type, dim*3, xyz*3, yaw
    #         # split_id.shape [bs]
    #         batch_indices = random_indices[i_batch:i_batch + N_rand]
    #         rays_batch = rays[:, batch_indices, :] # [bs, N_rand, 11]


    #         t0 = time.time()
    #         print(t0-last_t, len(split_id), i_batch)
    #         last_t = t0

    #         i_batch += N_rand
    #         if i_batch >= rays.shape[1]:
    #             print('Shuffle data after an epoch!')
    #             random_indices = torch.randperm(rays.shape[1])
    #             i_batch = 0

