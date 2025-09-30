import os
import numpy as np
import imageio.v2 as imageio
import torch


start_frame = 65
total_frame = 270
height = 375
width = 1242

img2mse = lambda x, y : np.mean((x - y)**2)
mse2psnr = lambda x : 10 * np.log10((255 ** 2) / x)
# img2mse = lambda x, y : torch.mean((x - y) ** 2)
# mse2psnr = lambda x : 10 * torch.log10(255 ** 2 / (x))
#mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
    
    
#img_raw = imageio.imread(os.path.join(suds_res_path, filename))
img_raw = imageio.imread('weights/opti_2w/renderonly_path_100000/020_0_0_2759.png')
H,W,_=img_raw.shape
h=H//2
gt = img_raw[:h,:, :]
pred = img_raw[h:, :, :]

#mask_path = os.path.join(mask_root, str(query_frame_id).zfill(6)+'_-1_mask.png')
mask_path = os.path.join('data/training/autorf/0006/000091_-1_mask.png')
mask = imageio.imread(mask_path).astype(bool)
#obj_gt = torch.from_numpy(gt[mask]).float()
#obj_pred = torch.from_numpy(pred[mask]).float()

obj_gt = gt[mask].astype(float)
obj_pred = pred[mask].astype(float)
psnr = mse2psnr(img2mse(obj_pred, obj_gt)).item()

print('psnr:', psnr)

    # tmp1 = gt.copy()
    # tmp1[~mask] = 0
    # tmp2 = pred.copy()
    # tmp2[~mask] = 0
    # imageio.imwrite('tmp.png', np.vstack([tmp1, tmp2]))
    # break