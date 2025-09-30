import numpy as np

class Object3d():
    def __init__(self, label_line):
        data = label_line.split(' ')
        data[:2] = [int(x) for x in data[:2]]
        data[3:] = [float(x) for x in data[3:]]

        self.img_id = data[0]  # 000000.png, 000001.png, ...
        self.obj_id = data[1]  # obj_0, obj_1, ... (same obj in different img)
        self.type = data[2] # 'Car', 'Truck', 'Van', 'DontCare', ...
        self.truncation = data[3] # truncated pixel ratio [0..1]
        self.occlusion = int(data[4]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[5] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[6] # left
        self.ymin = data[7] # top
        self.xmax = data[8] # right
        self.ymax = data[9] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = data[10] # box height
        self.w = data[11] # box width
        self.l = data[12] # box length (in meters)
        self.t = (data[13],data[14],data[15]) # location (x,y,z) in camera coord.
        self.dim = (self.l, self.h, self.w)
        self.ry = data[16] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]


class Calibration():
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_path):
        calibs = self.read_calib_file(calib_path)
        # Projection matrix from rect camera coord to image2/image3 coord
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])
        self.P3 = calibs['P3']
        self.P3 = np.reshape(self.P3, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = self.invert_trans(self.V2C)
        # Rigid transform from Imu coord to reference camera coord
        self.I2V = calibs['Tr_imu_velo']
        self.I2V = np.reshape(self.I2V, [3, 4])
        self.V2I = self.invert_trans(self.I2V)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P2[0,2]
        self.c_v = self.P2[1,2]
        self.f_u = self.P2[0,0]
        self.f_v = self.P2[1,1]
        self.b_x = self.P2[0,3]/(-self.f_u) # relative 
        self.b_y = self.P2[1,3]/(-self.f_v)

        tmp2, tmp3 = np.eye(4), np.eye(4)
        tmp2[:3, 3] = np.linalg.inv(self.P2[:, :3])@self.P2[:, 3]
        self.camrect2cam2 = tmp2
        tmp3[:3, 3] = np.linalg.inv(self.P3[:, :3])@self.P3[:, 3]
        self.camrect2cam3 = tmp3

    def read_calib_file(self, path):
        data = {}
        with open (path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.rstrip()
                if len(line)==0: continue
                if i < 4:
                    key, value = line.split(':')
                else:
                    key, value = line.split(' ', maxsplit=1)
                data[key] = np.array([float(x) for x in value.split()])
        return data
    
    def invert_trans(self, Tr):
        ''' Inverse a rigid body transform matrix [R'|-R't] (3x4 as [R|t])
        '''
        inv_Tr = np.zeros_like(Tr) # 3x4
        inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
        inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
        return inv_Tr
    
    def compute_label_03(self, obj, sh):

        bbox_3d_cam2 = np.array([
            [obj.l/2, obj.l/2, -obj.l/2, -obj.l/2, obj.l/2, obj.l/2, -obj.l/2, -obj.l/2],
            [0, 0, 0, 0, -obj.h, -obj.h, -obj.h, -obj.h],
            [obj.w/2, -obj.w/2, -obj.w/2, obj.w/2, obj.w/2, -obj.w/2, -obj.w/2, obj.w/2]
        ])
        bbox_ry_cam2 = np.array([
            [np.cos(obj.ry), 0, np.sin(obj.ry)],
            [0, 1, 0],
            [-np.sin(obj.ry), 0, np.cos(obj.ry)]
        ])
        bbox_3d_cam2 = bbox_ry_cam2@bbox_3d_cam2
        bbox_3d_cam2 += np.array(obj.t)[:, None]  # shape (3, 8)

        bbox_3d_rect = self.R0@bbox_3d_cam2
        bbox_3d_cam3 = self.P3@np.vstack((bbox_3d_rect, np.ones([1, bbox_3d_rect.shape[1]])))
        bbox_3d_cam3 /= bbox_3d_cam3[2, :]

        bbox_2d_cam3 = np.array([
            np.min(bbox_3d_cam3[0, :]),
            np.min(bbox_3d_cam3[1, :]),
            np.max(bbox_3d_cam3[0, :])+10,
            np.max(bbox_3d_cam3[1, :])+10
        ]).astype(np.int32)  # XYXY format
        bbox_2d_cam3[0] = np.max([0, bbox_2d_cam3[0]])
        bbox_2d_cam3[1] = np.max([0, bbox_2d_cam3[1]])
        bbox_2d_cam3[2] = np.min([sh[1], bbox_2d_cam3[2]])
        bbox_2d_cam3[3] = np.min([sh[0], bbox_2d_cam3[3]])

        return bbox_2d_cam3


def rotx(t):
    ''' Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def get_rotation(roll, pitch, heading):
    return rotz(heading)@roty(pitch)@rotx(roll)


def invert_transformation(Tr):
    ''' input 3x4 or 4x4, output 4x4
    '''
    inv_Tr = np.eye(4)
    inv_Tr[:3, :3] = Tr[:3, :3].T
    inv_Tr[:3, 3] = -Tr[:3, :3].T @ Tr[:3, 3]
    return inv_Tr