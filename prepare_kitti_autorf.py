import torch
import os
import numpy as np
from collections import defaultdict
import imageio.v2 as imageio
from tqdm import tqdm

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

if __name__ == '__main__':

    from segment_anything import sam_model_registry, SamPredictor

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    root_dir = './data/training/'
    target_types = ['Car', 'Van', 'Truck']

    label_root = os.path.join(root_dir, 'label_02')
    image_root = os.path.join(root_dir, 'image_02')
    for target_type in target_types:
        save_root = os.path.join(root_dir, 'autorf_dataset_'+target_type)
        for seq_number in sorted(os.listdir(label_root)):
            # seq_number: 0000.txt, 0001.txt, ...
            if not seq_number.endswith('.txt'):
                continue
            seq = seq_number.split('.')[0]
            save_path = os.path.join(save_root, seq)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(label_root, seq_number), 'r') as f:
                label_lines = f.readlines()
            objs_all_raw = [Object3d(line) for line in label_lines]

            obj_metadata = defaultdict(list)
            total_num = 0
            for obj in objs_all_raw:
                if obj.obj_id < 0 or obj.type != target_type or obj.truncation > 1.0 or obj.occlusion > 1:
                    continue

                k = obj.img_id
                v = (obj.obj_id, obj.box2d.astype(np.int32))
                obj_metadata[k].append(v)
                total_num += 1
            print('In seq', seq, ', there are', total_num, target_type, 'objs.') 
            pbar = tqdm(total=total_num, desc='seq_'+seq)

            for img_id, objs in obj_metadata.items():

                img_filename = os.path.join(image_root, seq, str(img_id).zfill(6)+'.png')
                img = imageio.imread(img_filename)

                predictor.set_image(img)

                for obj in objs:
                    obj_id = obj[0]
                    input_box = obj[1]

                    skip = False
                    for saved_filenames in os.listdir(save_path):
                        if '{:06d}_{:02d}'.format(img_id, obj_id) in saved_filenames:
                            skip = True
                            break
                    if skip:
                        continue

                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                    rgb_gt = img[input_box[1]:input_box[3], input_box[0]:input_box[2], :]
                    # the foreground is set to 1, and the background is set to 0
                    msk_gt = masks[0, input_box[1]:input_box[3], input_box[0]:input_box[2]]
                    msk_gt = (255 * msk_gt).astype(np.uint8)
                    msk_gt = np.repeat(msk_gt[:, :, None], 3, axis=-1)
                    imageio.imwrite(save_path+'/{:06d}_{:02d}_patch.png'.format(img_id, obj_id), rgb_gt)
                    imageio.imwrite(save_path+'/{:06d}_{:02d}_mask.png'.format(img_id, obj_id), msk_gt)

                    pbar.update(1)
            
            pbar.close()










        


                


