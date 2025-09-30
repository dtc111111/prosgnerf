import torch
from math import pi

def manipulation_obj_pose(obj, manipulation, **kwargs):
    # obj.shape: [1, max_obj, 11]
    #   frame_id, cam_id, obj_id, obj_type, dim*3, xyz*3, yaw
    if manipulation == 'rotate':
        steps = 11
        min = - pi/30
        max =   pi/30
        rotated_obj = obj.repeat(steps, 1, 1)  # [steps, max_obj, 11]
        rotated_obj[:, :, 10] += torch.linspace(min, max, steps, dtype=torch.float32, device=obj.device)[:, None]
        return rotated_obj
    elif manipulation == 'move_x':
        steps = 7
        min = - 3
        max =   3
        moved_obj = obj.repeat(steps, 1, 1)  # [steps, max_obj, 11]
        moved_obj[:, :, 7] += torch.linspace(min, max, steps, dtype=torch.float32, device=obj.device)[:, None] / kwargs['pose_scale_factor']
        return moved_obj
    elif manipulation == 'move_z':
        steps = 5
        min = - 2
        max =   2
        moved_obj = obj.repeat(steps, 1, 1)  # [steps, max_obj, 11]
        moved_obj[:, :, 9] += torch.linspace(min, max, steps, dtype=torch.float32, device=obj.device)[:, None] / kwargs['pose_scale_factor']
        return moved_obj
    elif manipulation == 'switch_location':
        all_objs = kwargs['visible_objects']
        num_img, num_obj, _ = all_objs.shape
        all_valid_objs = []
        for id_obj in range(num_obj):
            valid_objs_i = torch.stack([all_objs[id_img, id_obj, :] for id_img in range(num_img) if all_objs[id_img, id_obj, 0] > 0])
            all_valid_objs.append(valid_objs_i)
        new_obj = torch.zeros_like(obj)
        for id_obj in range(num_obj):
            random_i = torch.randint(0, len(all_valid_objs[id_obj]), (), dtype=torch.int64, device=obj.device)
            new_obj[0, id_obj] = all_valid_objs[id_obj][random_i]
        return new_obj
    else:
        raise NotImplementedError

def manipulation_cam_pose(pose, manipulation, **kwargs):
    # pose.shape: [1, 4, 4]
    if manipulation == 'move_x':
        steps = 5
        min = - 1
        max =   1
        moved_pose = pose.repeat(steps, 1, 1)
        moved_pose[:, 0, 3] += torch.linspace(min, max, steps, dtype=torch.float32, device=pose.device) / kwargs['pose_scale_factor']
        return moved_pose
    elif manipulation == 'move_y':
        steps = 5
        min = - 0.5
        max =   0.5
        moved_pose = pose.repeat(steps, 1, 1)
        moved_pose[:, 1, 3] += torch.linspace(min, max, steps, dtype=torch.float32, device=pose.device) / kwargs['pose_scale_factor']
        return moved_pose
    elif manipulation == 'move_z':
        steps = 5
        min = - 1
        max =   1
        moved_pose = pose.repeat(steps, 1, 1)
        moved_pose[:, 2, 3] += torch.linspace(min, max, steps, dtype=torch.float32, device=pose.device) / kwargs['pose_scale_factor']
        return moved_pose
    else:
        raise NotImplementedError
