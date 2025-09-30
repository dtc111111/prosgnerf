import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
import os
import copy
from collections import defaultdict
import torchvision.transforms as T
import matplotlib
from piqa.psnr import PSNR
from piqa.ssim import SSIM
from piqa.lpips import LPIPS

from tqdm import tqdm
from math import pi
from torchvision.models import resnet34, ResNet34_Weights
from mlp_res import MlpResNet

def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None):
    #   '''
    # Returns a frequency mask for position encoding in NeRF.
    
    # Args:
    #     pos_enc_length (int): Length of the position encoding.
    #     current_iter (int): Current iteration step.
    #     total_reg_iter (int): Total number of regularization iterations.
    #     max_visible (float, optional): Maximum visible range of the mask. Default is None. 
    #     For the demonstration study in the paper.
        
    #     Correspond to FreeNeRF paper:
    #     L: pos_enc_length
    #     t: current_iter
    #     T: total_iter
    
    # Returns:
    #     jnp.array: Computed frequency or visibility mask.
    # '''
    if max_visible is None:
        # default FreeNeRF
        if current_iter < total_reg_iter:
            freq_mask = torch.zeros(pos_enc_length)  # all invisible  
            ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1
            ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
            int_ptr = int(ptr)
            freq_mask[:int_ptr*3] = 1.0  # assign the integer part
            freq_mask[int_ptr*3:int_ptr*3+3] = (ptr - int_ptr) # assign the fractional part
            return torch.clamp(freq_mask, 1e-8, 1-1e-8) # for numerical stability
        else:
            return torch.ones(pos_enc_length)
    
    else:
        # For the ablation study
        freq_mask = torch.zeros(pos_enc_length)
        freq_mask[:int(pos_enc_length * max_visible)] = 1.0
        return freq_mask

def lossfun_occ_reg(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    
    # Compute mean RGB value 
    rgb_mean = rgb.mean(-1)
    
    # Compute mask for white/black background
    if wb_prior:
        white_mask = torch.where(rgb_mean > 0.99, torch.ones_like(rgb_mean), torch.zeros_like(rgb_mean))
        black_mask = torch.where(rgb_mean < 0.01, torch.ones_like(rgb_mean), torch.zeros_like(rgb_mean))
        rgb_mask = white_mask + black_mask
        rgb_mask[:, wb_range:] = 0
    else:
        rgb_mask = torch.zeros_like(rgb_mean)
        
    # Create regularization mask    
    if reg_range > 0:
        rgb_mask[:, :reg_range] = 1
        
    # Compute weighted loss
    return torch.mean(density * rgb_mask)

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
img_transform = T.Compose([T.Resize((80, 120)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class LatentCodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        # (batchsize, feature_dimension(512))
        output = F.max_pool2d(latents, kernel_size=latents.size()[2:])[:, :, 0, 0]
        
        return output

def convert_batch_norm(layer, new_norm="instance"):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    if new_norm == "group":
                        layer._modules[name] = torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)
                    elif new_norm == "instance":
                        layer._modules[name] = torch.nn.InstanceNorm2d(num_channels)

            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_batch_norm(sub_layer, new_norm)
                layer.__setattr__(name=name, value=sub_layer)
    return layer

class MultiHeadImageEncoder(nn.Module):
    def __init__(self, color_size=128, density_size=128):
        super().__init__()
        backbone_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        backbone_model = convert_batch_norm(backbone_model)
        self.shared_model = nn.ModuleDict([
            ['conv1', backbone_model.conv1],
            ['bn1', backbone_model.bn1],
            ['relu', backbone_model.relu],
            ['maxpool', backbone_model.maxpool],
            ['layer1', backbone_model.layer1],
            ['layer2', backbone_model.layer2],
            ['layer3', backbone_model.layer3]
        ])
        self.color_head = nn.Sequential(
            copy.deepcopy(backbone_model.layer4),
            copy.deepcopy(backbone_model.avgpool),
            nn.Linear(512, color_size)
        )
        self.density_head = nn.Sequential(
            copy.deepcopy(backbone_model.layer4),
            copy.deepcopy(backbone_model.avgpool),
            nn.Linear(512, density_size)
        )
    
    def head_forward(self, head_model, x):
        for head_idx, head_layer in enumerate(head_model):
            if head_idx == len(head_model) - 1:
                x = torch.flatten(x, 1)
                x = head_layer(x)
            else:
                x = head_layer(x)
        return x
    
    def forward(self, x):
        out = {}
        x = self.shared_model.conv1(x)
        x = self.shared_model.bn1(x)
        x = self.shared_model.relu(x)

        x = self.shared_model.maxpool(x)
        x = self.shared_model.layer1(x)

        x = self.shared_model.layer2(x)

        x = self.shared_model.layer3(x)

        out['color'] = self.head_forward(self.color_head, x)
        out['density'] = self.head_forward(self.density_head, x)

        return out

class ConditionalRenderer(nn.Module):
    def __init__(self, use_embed_viewdirs = False, use_obj_pose = False,
                 code_size = 128):

        super().__init__()

        self.embed_param = {
            'num_freqs': 6,
            'num_freqs_views': 4,
            'num_freqs_objpose': 4,
            'freq_factor': 1.5,
            'include_input': True
        }
        self.use_embed_viewdirs = use_embed_viewdirs
        ###### TODO
        self.use_obj_pose = use_obj_pose
        self.dim_embed = 3 * self.embed_param['num_freqs'] * 2
        self.dim_embed += 3 if self.embed_param['include_input'] else 0
        self.dim_embed += 3 * self.embed_param['num_freqs_views'] * 2 if self.use_embed_viewdirs \
                            else 0
        self.dim_embed += 3 if self.use_embed_viewdirs and self.embed_param['include_input'] \
                            else 0
        self.dim_embed += 3 * self.embed_param['num_freqs_objpose'] * 2 if self.use_obj_pose \
                            else 0
        self.dim_embed += 3 if self.use_obj_pose and self.embed_param['include_input'] \
                            else 0
        self.code_size = code_size

        density_mlp_param = {
            'code_size': self.code_size,
            'num_layer': 5,
            'd_hidden': 128,
            'd_out': 1,
            'add_out_lvl': 3,
            'agg_fct': 'mean'
        }  # add_out_lvl: additional out level
        density_injections = {
            0 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            1 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            2 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            3 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            4 : {"lat,pos" : [self.code_size + self.dim_embed, 128]}
        }  # self.code_size + self.dim_embed = 167 = 128 + 3*13 = 128 + 3 * (2 * 6 + 1)

        color_mlp_param = {
            'code_size': self.code_size,
            'num_layer': 5,
            'd_hidden': 128,
            'd_out': 3,
            'agg_fct': 'mean'
        }
        color_injections = {
            0 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            1 : {"lat,pos" : [self.code_size + self.dim_embed, 128]},
            2 : {"lat,pos" : [self.code_size + self.dim_embed, 128],
                "density_feats": [128, 128]},
            3 : {"lat,pos" : [self.code_size + self.dim_embed, 128],
                'query_cams' : [3, 128]},
            4 : {"lat,pos" : [self.code_size + self.dim_embed, 128],
                'query_cams' : [3, 128]}
        }  # query_cams: viewdirs

        self.density_mlp = MlpResNet(
            d_in=density_mlp_param['code_size'] + self.dim_embed,
            dims=density_mlp_param['num_layer'] * [density_mlp_param['d_hidden']],
            d_out=density_mlp_param['d_out'],
            injections=density_injections,
            agg_fct=density_mlp_param['agg_fct'],
            add_out_lvl=density_mlp_param['add_out_lvl']
        )

        self.color_mlp = MlpResNet(
            d_in=color_mlp_param['code_size'] + self.dim_embed,
            dims=color_mlp_param['num_layer'] * [color_mlp_param['d_hidden']],
            d_out=color_mlp_param['d_out'],
            injections=color_injections,
            agg_fct=color_mlp_param['agg_fct']
        )

    def embed(self, p, flag = 0):
        if flag == 0:
            L = self.embed_param['num_freqs']
        elif flag == 1:
            L = self.embed_param['num_freqs_views']
        else:
            L = self.embed_param['num_freqs_objpose']
        
        freq_factor = self.embed_param['freq_factor']

        embed_p = torch.cat([
            torch.cat(
            [torch.sin((freq_factor*2**i)*pi*p), torch.cos((freq_factor*2**i)*pi*p)], dim=-1
            ) for i in range(L)
        ], dim=-1)

        if self.embed_param['include_input']:
            embed_p = torch.cat((p, embed_p), dim=-1)
        
        return embed_p

    def forward(self, step, pts_raw, views_raw, latent, obj_pose_raw=None):
        query_points = self.embed(pts_raw, 0)
        if self.use_embed_viewdirs:
            query_cams = self.embed(views_raw, 1)
        else:
            query_cams = views_raw
        if self.use_obj_pose:
            obj_pose = self.embed(obj_pose_raw, 2)
            query_points = torch.cat([query_points, obj_pose], 1)

        density_lat = latent['density']
        density_inj = {
            "lat,pos": torch.cat([density_lat, query_points], 1),
            "lat": density_lat,
            "query_points": query_points,
            "query_cams": query_cams
        }
        density_input = torch.cat([density_lat, query_points], 1)
        sigma_out, density_feats = self.density_mlp(density_input, inj_data=density_inj)

        color_lat = latent['color']
        color_inj = {
            "lat,pos": torch.cat([color_lat, query_points], 1),
            "lat": color_lat,
            "query_points": query_points,
            "query_cams": query_cams,
            "density_feats": density_feats
        }
        color_input = torch.cat([color_lat, query_points], 1)
        rgb_out = self.color_mlp(color_input, inj_data=color_inj)

        return torch.cat([rgb_out, sigma_out], -1), density_feats


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, D_views=4, skips=[4], latent_sz=256, 
                 n_freq_posenc=10, n_freq_posenc_views=4, n_freq_posenc_obj=4,
                 use_original_latents=False, total_step=50000):

        super().__init__()
        self.D = D
        self.W = W
        self.D_views = D_views
        self.skips = skips
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.n_freq_posenc_obj = n_freq_posenc_obj
        self.use_original_latents = use_original_latents
        self.total_step = total_step
        
        dim_embed = 3 * (self.n_freq_posenc * 2 + 1) # xyz
        dim_embed_view = 3 * (self.n_freq_posenc_views * 2 + 1) # dir
        dim_embed_obj = 3 * (self.n_freq_posenc_obj * 2 + 1) # obj pose

        # Density Prediction Layers   block 1
        self.pts_in = nn.Linear(dim_embed, W)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(dim_embed + latent_sz, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(dim_embed + latent_sz + W, W) for i in range(D - 1)]
        )
        self.pts_linears_bckg = nn.ModuleList(
            [nn.Linear(dim_embed, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(dim_embed + W, W) for i in range(D - 1)]
        )
        self.pts_linears_nsg = nn.ModuleList(
            [nn.Linear(dim_embed + latent_sz, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(dim_embed + latent_sz + W, W) for i in range(D - 1)]
        )

        self.sigma_out = nn.Linear(W, 1)  # volume density

        # Feature Prediction Layers   block 2
        self.feature_linear = nn.Linear(W, W)
        self.views_in = nn.Linear(dim_embed_view, W)

        self.views_linears = nn.ModuleList(
            [nn.Linear(W + dim_embed_view + latent_sz + dim_embed_obj, W//2)] + 
            [nn.Linear(W//2, W//2) for _ in range(D_views - 1)]
        )
        self.views_linears_bckg = nn.ModuleList(
            [nn.Linear(W + dim_embed_view, W//2)] + 
            [nn.Linear(W//2, W//2) for _ in range(D_views - 1)]
        )
        self.views_linears_nsg = nn.ModuleList(
            [nn.Linear(W + dim_embed_view + dim_embed_obj, W//2)] + 
            [nn.Linear(W//2, W//2) for _ in range(D_views - 1)]
        )

        self.latent_layer = nn.Sequential(nn.Linear(512, latent_sz), nn.ReLU())
        self.latent_view_layer = nn.Sequential(nn.Linear(512, latent_sz), nn.ReLU())

        self.rgb_out = nn.Linear(W//2, 3)

    def embed(self, step, p, flag = 0):
        if flag == 0:
            L = self.n_freq_posenc
        elif flag == 1:
            L = self.n_freq_posenc_views
        else:
            L = self.n_freq_posenc_obj
        pose_encoded=[torch.cat(
            [torch.sin((2**i)*pi*p), torch.cos((2**i)*pi*p)], dim=-1
            ) for i in range(L)]
        
        pose_encoded = torch.cat([
            torch.cat([
                torch.sin((2**i)*pi*p), 
                torch.cos((2**i)*pi*p)
            ], dim=-1) for i in range(L)
        ],-1)
        if step < self.total_step and step > -1:
            freq_reg_mask = get_freq_reg_mask(pose_encoded.shape[-1],step,self.total_step).to(pose_encoded.device)
            pose_encoded = pose_encoded * freq_reg_mask
        
        return torch.cat([p] + [pose_encoded], dim=-1)

    def forward(self, step, pts_raw, views_raw, latent=None, obj_pose_raw=None):
        latent_shape = None
        pts = self.embed(step, pts_raw, flag=0)
        if latent is not None:
            if self.use_original_latents:
                input = torch.cat([pts, latent], dim=-1)
                latent_shape = latent
                net1 = self.pts_linears_nsg
            else:
                latent_shape = self.latent_layer(latent)
                latent_app = self.latent_view_layer(latent)
                input = torch.cat([pts, latent_shape], dim=-1)
                net1 = self.pts_linears
        else:
            input = pts
            net1 = self.pts_linears_bckg
        h = input

        for i, layer in enumerate(net1):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([input, h], dim=-1)

        sigma = self.sigma_out(h)#[661856,1]
        feature = self.feature_linear(h)  # output of block 1
        input_views = self.embed(step, views_raw, flag=1)
        if obj_pose_raw is not None:
            obj_pose = self.embed(step, obj_pose_raw, flag=2)
            if self.use_original_latents:
                h = torch.cat([feature, input_views, obj_pose], -1)
                net2 = self.views_linears_nsg
            else:
                h = torch.cat([feature, input_views, latent_app, obj_pose], -1)
                net2 = self.views_linears
        else:
            h = torch.cat([feature, input_views], dim=-1)
            net2 = self.views_linears_bckg
    
        for i, layer in enumerate(net2):
            h = F.relu(layer(h))

        rgb = self.rgb_out(h)#[661856,3]
        outputs = torch.cat([rgb, sigma], -1)

        return outputs, latent_shape


def get_latents_all(img2_root, encoder, device):
    sequence = img2_root[-4:]
    data_root = img2_root.split('image_02')[0]
    mask_root = os.path.join(data_root, 'autorf', sequence)

    batch_size = 24

    latents_all = {}
    img_patches = []
    tmp_mask = None
    ids = []

    for file_name in sorted(os.listdir(mask_root)):
        if not file_name.endswith('.png'):
            continue
        splited = file_name.split('_')
        frame_id = int(splited[0])
        obj_id = int(splited[1])
        img_type = splited[-1]
        # maskes and patches are stored in sort
        if img_type == 'mask.png':
            tmp_mask = imageio.imread(os.path.join(mask_root, file_name)).astype(bool)
        elif img_type == 'patch.png':
            tmp_patch = imageio.imread(os.path.join(mask_root, file_name))
            tmp_patch[~tmp_mask] = 0
            img_patches.append(tmp_patch)
            ids.append((frame_id, obj_id))

    for i in tqdm(range(0, len(img_patches), batch_size)):
        batch = img_patches[i:i+batch_size]
        id_batch = ids[i:i+batch_size]
        # TODO
        img_batch = [img_transform(T.ToTensor()(img_patch))[None, ...] for img_patch in batch]
        img_batch = torch.cat(img_batch).to(device)
        latents = encoder(img_batch)
        for j, (frame_id, obj_id) in enumerate(id_batch):
            if 'MultiHead' in encoder._get_name():
                latents_all[(frame_id, obj_id)] = {
                    'color': latents['color'][j][None, ...].detach(),
                    'density': latents['density'][j][None, ...].detach()
                }
            else:
                latents_all[(frame_id, obj_id)] = latents[j][None, ...].detach()
        if 'MultiHead' in encoder._get_name():
            latents_all['MultiHead'] = True
        else:
            latents_all['MultiHead'] = False

    return latents_all


def get_latents_all_nsg(latent_sz, scene_objects, device):
    '''
    the original nsg implement of latents
    '''
    
    latents_all = {int(obj_id): nn.Parameter(torch.empty(1, latent_sz, device=device).normal_(mean=0., std=0.01), \
                                        requires_grad=True) for obj_id in scene_objects}

    return latents_all

# # Compute segment loss on rendered images
# def get_render_segment_loss(imgs, relative_frame_ids, img2_root, index_list, savedir, device):
#     # sam
#     from segment_anything import sam_model_registry, SamPredictor
#     sam_checkpoint = "sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     predictor = SamPredictor(sam)
#     # mask
#     from prepare_kitti import Calibration, Object3d
#     sequence = img2_root[-4:]
#     data_root = img2_root.split('image_02')[0]
#     img3_root = os.path.join(os.path.join(data_root, 'image_03'), sequence)
#     calib_path = os.path.join(os.path.join(data_root, 'calib'), sequence+'.txt')
#     label_path = os.path.join(os.path.join(data_root, 'label_02'), sequence+'.txt')
#     mask_root = os.path.join(os.path.join(data_root, 'autorf'), sequence)

#     frame_ids = [index_list[x] for x in relative_frame_ids]

#     calib = Calibration(calib_path)

#     with open(label_path, 'r') as read_label:
#         label_lines = read_label.readlines()
#     objs = defaultdict(list)
#     frame_num = len(os.listdir(img2_root))
#     for line in label_lines:
#         label_frame_id = int(line.split()[0])
#         if label_frame_id in frame_ids:
#             objs[label_frame_id].append(Object3d(line))
#         elif label_frame_id+frame_num in frame_ids:
#             objs[label_frame_id+frame_num].append(Object3d(line))


#     mse_all = 0
#     imgs = (imgs*255).astype(np.uint8)
#     for i in range(len(imgs)):
#         predictor.set_image(imgs[i])
#         frame_id = frame_ids[i]
#         is_cam2 = frame_id < frame_num
#         whole_mask = np.zeros(imgs.shape[1:3], dtype=bool)
#         for obj in objs[frame_id]:
#             if obj.type == 'DontCare':
#                 continue
#             if is_cam2:
#                 input_box = obj.box2d.astype(np.int32)
#             else:
#                 input_box = calib.compute_label_03(obj, imgs[i].shape)
#             masks, _, _ = predictor.predict(
#                 point_coords=None,
#                 point_labels=None,
#                 box=input_box[None, :],
#                 multimask_output=False,
#             )
#             whole_mask += masks[0]
#         gt_mask = imageio.imread(mask_root+'/{:06d}_-1_mask.png'.format(frame_id))[:, :, 0].astype(bool)
#         img2save = np.vstack((np.stack([gt_mask*255]*3, -1), np.stack([whole_mask*255]*3, -1)))
#         imageio.imwrite(savedir+'/{:06}_mask.png'.format(frame_id), img2save)
#         mse = np.mean(gt_mask^whole_mask)
#         mse_all += mse
#     mse_all /= len(imgs)
#     return mse_all

def metrics(pred_double, gt, savefile=None, lpips_network='vgg'):
    device = pred_double.device
    pred = pred_double.to(torch.float32)
    pred_trans = pred.transpose(0, 2)[None, ...]
    gt_trans = gt.transpose(0, 2)[None, ...]
    psnr_func = PSNR().to(device)
    psnr = psnr_func(pred_trans, gt_trans)
    ssim_func = SSIM().to(device)
    ssim = ssim_func(pred_trans, gt_trans)
    lpips_func = LPIPS(network='vgg').to(device)
    lpips_vgg = lpips_func(pred_trans, gt_trans)
    lpips_func = LPIPS(network='alex').to(device)
    lpips_alex = lpips_func(pred_trans, gt_trans)
    if savefile is not None:
        with open(savefile, 'a') as f:
            f.write(f'{psnr.item()} {ssim.item()} {lpips_vgg.item()} {lpips_alex.item()}\n')
    return psnr.item(), ssim.item(), lpips_vgg.item(), lpips_alex.item()

class ProposalMLP(nn.Module):
    def __init__(self, D=4, W=256, n_freq_posenc=10):
        super().__init__()
        
        self.n_freq_posenc = n_freq_posenc
        dim_embed = 3 * (self.n_freq_posenc * 2 + 1) # xyz
        self.pts_linears = nn.ModuleList(
            [nn.Linear(dim_embed, W)] + 
            [nn.Linear(W, W) for _ in range(D - 1)]
        )
        for module in self.pts_linears:
            nn.init.kaiming_uniform_(module.weight)
        self.sigma_out = nn.Linear(W, 1)
        nn.init.kaiming_uniform_(self.sigma_out.weight)

    def embed(self, p):
        L = self.n_freq_posenc
        
        return torch.cat([p] + [
            torch.cat(
            [torch.sin((2**i)*pi*p), torch.cos((2**i)*pi*p)], dim=-1
            ) for i in range(L)
        ], dim=-1)

    def forward(self, pts_raw):
        pts = self.embed(pts_raw)
        h = pts
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
        h = self.sigma_out(h)
        return h

def UniformLinDispPiecewiseSample(num_samples, near, far, perturb=0.0):
    spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
    spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x))
    bins = torch.linspace(0.0, 1.0, num_samples + 1, device=near.device)[None, ...]  # [1, num_samples+1]

    if perturb > 0:
        t_rand = torch.rand_like(near)
        bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
        bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
        bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
        bins = bin_lower + (bin_upper - bin_lower) * t_rand
    else:
        bins = bins.repeat(near.shape[0], 1)

    s_near, s_far = (spacing_fn(x) for x in (near, far))
    spacing_to_euclidean_fn = lambda x: spacing_fn_inv(x * s_far + (1 - x) * s_near)
    euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

    return euclidean_bins, bins, spacing_to_euclidean_fn

def PDFSample(num_samples, existing_bins, spacing_to_euclidean_fn, weights, perturb=0.0):
    # existing_bins in s-space
    # weights.shape: N_rays, num_samples
    num_bins = num_samples + 1
    
    histogram_padding = 0.01
    weights = weights + histogram_padding
    # Add small offset to rays with zero weight to prevent NaNs
    eps = 1e-5
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding

    pdf = weights / weights_sum
    cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if perturb > 0:
        # Stratified samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
        rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
        u = u + rand
    else:
        # Uniform samples between 0 and 1
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u + 1.0 / (2 * num_bins)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, side="right")
    below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(existing_bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(existing_bins, -1, above)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    bins = bins_g0 + t * (bins_g1 - bins_g0)

    bins = bins.detach()
    euclidean_bins = spacing_to_euclidean_fn(bins)

    return euclidean_bins, bins

def sample_near_depth(gt_depth, near, far, ratio = 0.1):
    depth_mask = gt_depth > 0
    near_tensor = torch.ones_like(gt_depth) * near
    far_tensor = torch.ones_like(gt_depth) * far
    near_tensor[depth_mask] = gt_depth[depth_mask] * (1-ratio)
    far_tensor[depth_mask] = gt_depth[depth_mask] * (1+ratio)
    return near_tensor.unsqueeze(-1), far_tensor.unsqueeze(-1)

class NearFarCollider(nn.Module):
    def __init__(self, near, far, scene_bounds, device):
        super().__init__()
        self.device = device
        self.near = torch.tensor(near, device=self.device)
        self.far = torch.tensor(far, device=self.device)
        self.scene_bounds = torch.tensor(scene_bounds, device=self.device)  # 2 x 3
        self.sphere_center = (self.scene_bounds[1] + self.scene_bounds[0]) * 0.5
        self.sphere_radius = (self.scene_bounds[1] - self.scene_bounds[0]) * np.sqrt(3) / 2
    
    def forward(self, rays_o, rays_d):
        nears = self.near.expand([len(rays_o), 1])
        fars = self.far.expand([len(rays_o), 1])
        
        nears = self.truncate_with_plane_intersection(rays_o, rays_d, self.scene_bounds[1, 2], nears)
        fars = self.truncate_with_plane_intersection(rays_o, rays_d, self.scene_bounds[0, 2], fars)

        rays_o_sphere, rays_d_sphere = self.ellipse_to_sphere_coords(rays_o, rays_d)
        _, sphere_fars = self.intersect_with_sphere(rays_o_sphere, rays_d_sphere, torch.zeros(3, device=self.device))
        fars = torch.minimum(fars, sphere_fars)
        
        assert nears.min()>=0 and fars.min()>=0
        
        nears = nears.clamp_min(self.near)
        fars = fars.clamp_min(nears + 1e-6).clamp_max(self.far)

        return nears.to(torch.float32), fars.to(torch.float32)

    def truncate_with_plane_intersection(self, rays_o, rays_d, altitude, default_bounds):
        starts_before = rays_o[..., 2] > altitude
        goes_down = rays_d[..., 2] < 0

        boundable_rays = torch.minimum(starts_before, goes_down)
        new_bounds = default_bounds.clone()

        ray_points = rays_o[boundable_rays]
        ray_dirs = rays_d[boundable_rays]
        if ray_points.shape[0] == 0:
            return new_bounds

        new_bounds[boundable_rays] = ((altitude - ray_points[..., 2]) / ray_dirs[..., 2]).unsqueeze(-1)

        return new_bounds
    
    def ellipse_to_sphere_coords(self, rays_o, rays_d):
        rays_o_sphere = (rays_o - self.sphere_center) / self.sphere_radius
        rays_d_sphere = rays_d / self.sphere_radius
        return rays_o_sphere, rays_d_sphere

    def intersect_with_sphere(self, rays_o, rays_d, center, radius=1.0, near_plane=0.0):
        a = (rays_d * rays_d).sum(dim=-1, keepdim=True)
        b = 2 * (rays_o - center) * rays_d
        b = b.sum(dim=-1, keepdim=True)
        c = (rays_o - center) * (rays_o - center)
        c = c.sum(dim=-1, keepdim=True) - radius**2

        # clamp to near plane
        nears = (-b - torch.sqrt(torch.square(b) - 4 * a * c)) / (2 * a)
        fars = (-b + torch.sqrt(torch.square(b) - 4 * a * c)) / (2 * a)

        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)

        nears = torch.nan_to_num(nears, nan=0.0)
        fars = torch.nan_to_num(fars, nan=0.0)

        return nears, fars


def get_weights(raw, z_vals, rays_d=None, raw_noise_std=0, use_rays_d=False):
    # z_vals: [N_rays, N_samples + 1]

    if raw_noise_std > 0.:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std
    else:
        noise = 0.
    if use_rays_d:
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samplees]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    else:
        weights = get_weights_from_density(z_vals, raw[..., 3] + noise)
    
    return weights

def get_weights_from_density(z_vals, density):
    # z_vals.shape: N_rays x (N_sample + 1)
    # density.shape: N_rays x N_sample x 1
    # deltas.shape: N_rays x N_sample,
    deltas = z_vals[..., 1:] - z_vals[..., :-1]
    deltas = torch.cat([torch.zeros_like(z_vals[..., :1]), deltas], dim=-1)
    density_trans = F.softplus(density - 1).squeeze()
    delta_density = deltas * density_trans
    alphas = 1 - torch.exp(-delta_density)
    
    
    transmittance = torch.cumsum(delta_density[..., :-1], dim=-1)
    transmittance = torch.cat([torch.zeros_like(transmittance[..., :1]), transmittance], dim=-1)
    transmittance = torch.exp(-transmittance)

    weights = alphas * transmittance
    weights = torch.nan_to_num(weights)

    return weights

def apply_colormap(img_np, cmap='plasma'):
    # Args: img_np [H, W]
    # Return: color [H, W, 3], 0-1
    img = img_np / img_np.max()
    colormap = matplotlib.colormaps[cmap].resampled(256)
    return colormap(img)[:, :, :3]

def outer(t0_starts, t0_ends, t1_starts, t1_ends, y1):
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(t, w, t_env, w_env):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + 1e-7)

def zvals_to_sdist(z_vals):
    spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x))
    sdist = spacing_fn(z_vals)
    return sdist

def interlevel_loss(weights_list, z_vals_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = zvals_to_sdist(z_vals_list[-1]).detach()
    w = weights_list[-1].detach()
    loss_interlevel = 0.0
    for z_vals, weights in zip(z_vals_list[:-1], weights_list[:-1]):
        sdist = zvals_to_sdist(z_vals)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, z_vals_list):
    """From mipnerf360"""
    c = zvals_to_sdist(z_vals_list[-1])
    w = weights_list[-1]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')  # ('yx')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(c2w.device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return torch.stack([rays_o, rays_d], dim=0)


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


# original NERF sampling method
def sample_along_ray(rays, near, far, N_samples, N_rays, method, perturb):
    rays_o, rays_d = rays
    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    if method == 'lindisp':
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    elif method == 'squareddist':
        pass
    else:
        z_vals = near * (1.-t_vals) + far * (t_vals)
        if method == 'discrete':
            perturb = 0
    
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand_like(z_vals)

        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples, 3]
    return pts, z_vals


def plane_pts(rays, planes, id_planes, near, method='planes'):
    """ Ray-Plane intersection for given planes in the scene

    Args:
        rays: ray origin and directions
        planes: first plane position, plane normal and distance between planes
        id_planes: ids of used planes
        near: distance between camera pose and first intersecting plane
        method: Method used

    Returns:
        pts: [N_rays, N_samples+N_importance] - intersection points of rays and selected planes
        z_vals: position of the point along each ray respectively
    """
    # Extract ray and plane definitions
    rays_o, rays_d = rays
    N_rays = rays_o.shape[0]
    plane_bds, plane_normal, delta = planes

    # Get amount of all planes
    n_planes = torch.ceil(torch.norm(plane_bds[:, -1] - plane_bds[:, 0]) / delta) + 1

    # Calculate how far the ray_origins lies apart from each plane
    d_ray_first_plane = torch.matmul(plane_bds[:, 0]-rays_o, plane_normal[:, None])
    d_ray_first_plane = torch.max(-d_ray_first_plane, -near)

    # Get the ids of the planes in front of each ray starting from near distance upto the far plane
    start_id = torch.ceil((d_ray_first_plane+near)/delta)
    plane_id = start_id + id_planes
    if method == 'planes':
        plane_id = torch.cat([plane_id[:, :-1], n_planes.repeat(N_rays)[:, None]], dim=1)
    elif method == 'planes_plus':
        # Experimental setup, that got discarded due to lower or the same quality
        plane_id = torch.cat([plane_id[:, :1],
                            id_planes[None, 1:-1].repeat(N_rays, 1),
                            n_planes.repeat(N_rays)[:, None]], dim=1)

    # [N_samples, N_rays, xyz]
    z_planes = plane_normal[None, None, :] * (plane_id*delta).transpose(0, 1)[..., None]
    relevant_plane_origins = plane_bds[:, 0][None, None, :]+z_planes

    # Distance between each ray's origin and associated planes
    d_plane_pose = relevant_plane_origins - rays_o[None, :, :]

    n = torch.matmul(d_plane_pose, plane_normal[..., None])
    z = torch.matmul(rays_d, plane_normal[..., None])

    z_vals = (n / z).squeeze().transpose(0 ,1)

    pts = rays_o[..., None ,:] + rays_d[..., None ,:] * z_vals[..., None]

    return pts, z_vals


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        p = p[..., None, :]

    c_y = torch.cos(yaw)[..., None]
    s_y = torch.sin(yaw)[..., None]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    return torch.cat([p_x[..., None], p_y[..., None], p_z[..., None]], dim=-1)


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = torch.tensor([1., 1., 1.]).to(sc_factor.device) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9))[:, :, None, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1/scaling_factor) * p

    return p_scaled


def world2object(pts, dirs, pose, theta_y, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    #  Prepare args if just one sample per ray-object or world frame only
    if pts.ndim == 3:
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = pose.repeat(n_sample_per_ray)
        theta_y = theta_y.repeat(n_sample_per_ray)
        if dim is not None:
            dim = dim.repeat(n_sample_per_ray)
        if dirs.ndim == 2:
            dirs = dirs.repeat(n_sample_per_ray)

        pts = pts.reshape(-1, 3)

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = (torch.tensor([0., -1., 0.])[None, :].to(dim.device) if inverse else
               torch.tensor([0., -1., 0.])[None, None, :]).to(dim.device) * \
              (dim[..., 1] / 2)[..., None]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    if not inverse:
        N_obj = theta_y.shape[1]
        pts_w = pts[:, None, :].repeat(1, N_obj, 1)
        dirs_w = dirs[:, None, :].repeat(1, N_obj, 1)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / torch.norm(dirs_o, dim=3)[..., None, :]
        return [pts_o, dirs_o]
    else:
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / torch.norm(dirs_w, axis=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def object2world(pts, dirs, pose, theta_y, dim=None, inverse=True):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in N_obj object frames, [N_pts, N_obj, 3]
        dirs: Corresponding 3D directions given in N_obj object frames, [N_pts, N_obj, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]

    Returns:
        pts_w: 3d points transformed into world frame
        dir_w: unit - 3d directions transformed into world frame
    """

    #  Prepare args if just one sample per ray-object
    if len(pts.shape) == 3:
        # [N_rays, N_obj, N_obj_samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = pose.repeat(n_sample_per_ray)
        theta_y = theta_y.repeat(n_sample_per_ray)
        if dim is not None:
            dim = dim.repeat(n_sample_per_ray)
        if len(dirs.shape) == 2:
            dirs = dirs.repeat(n_sample_per_ray)

        pts = pts.reshape(-1, 3)

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = torch.tensor([0., -1., 0.])[None, :].to(dim.device) * (dim[..., 1] / 2)[..., None]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    pts_o = pts[None, :, None, :]
    dirs_o = dirs
    if dim is not None:
        pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
        if dirs is not None:
            dirs_o = scale_frames(dirs_o, dim, inverse=True)

    pts_o = pts_o - t_w_o
    pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

    if dirs is not None:
        dirs_w = rotate_yaw(dirs_o, -theta_y)
        # Normalize direction
        dirs_w = dirs_w / torch.norm(dirs_w, axis=-1)[..., None, :]
    else:
        dirs_w = None

    return [pts_w, dirs_w]


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in: 1-D
        z_ray_out: 1-D
        # intersection_map: Maps intersection values in z to their ray-box intersection
        intersection_map: Boolean value, [rays, boxes]
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)

    inv_d = torch.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.min(t_min, t_max)
    t1 = torch.max(t_min, t_max)

    t_near = torch.max(torch.max(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.min(torch.min(t1[..., 0], t1[..., 1]), t1[..., 2])
    
    # Check if rays are inside boxes
    # a tensor with all bool, same shape as t_far
    # use larger space than torch.where(t_far > t_near)
    intersection_map = t_far > t_near  # [rays, boxes] [rays_batchsize, n_obj]
    # Check that boxes are in front of the ray origin
    positive_far = (t_far*intersection_map) > 0  # [rays, boxes]
    intersection_map = torch.bitwise_and(intersection_map, positive_far)  # [rays, boxes]

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map]  # 1-D
        z_ray_out = t_far[intersection_map]  # 1-D
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def box_pts(rays, pose, theta_y, dim=None, one_intersec_per_ray=False, mask_only=False):
    """gets ray-box intersection points in world and object frames in a sparse notation

    Args:
        rays: ray origins and directions, [[N_rays, 3], [N_rays, 3]]
        pose: object positions in world frame for each ray, [N_rays, N_obj, 3]
        theta_y: rotation of objects around world y axis, [N_rays, N_obj]
        dim: object bounding box dimensions [N_rays, N_obj, 3]
        one_intersec_per_ray: If True only the first interesection along a ray will lead to an
        intersection point output

    Returns:
        pts_box_w: box-ray intersection points given in the world frame
        viewdirs_box_w: view directions of each intersection point in the world frame
        pts_box_o: box-ray intersection points given in the respective object frame
        viewdirs_box_o: view directions of each intersection point in the respective object frame
        z_vals_w: integration step in the world frame
        z_vals_o: integration step for scaled rays in the object frame
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at the intersection

    """
    rays_o, rays_d = rays
    # Transform each ray into each object frame
    rays_o_o, dirs_o = world2object(rays_o, rays_d, pose, theta_y, dim)
    rays_o_o = rays_o_o.squeeze(-2)
    dirs_o = dirs_o.squeeze(-2)

    # Get the intersection with each Bounding Box
    z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(rays_o_o, dirs_o)

    if mask_only and not one_intersec_per_ray:
        return intersection_map

    if z_ray_in_o is not None and len(z_ray_in_o) != 0:
        # Calculate the intersection points for each box in each object frame
        pts_box_in_o = rays_o_o[intersection_map] + z_ray_in_o[:, None]*dirs_o[intersection_map]
        
        # Transform the intersection points for each box in world frame
        pts_box_in_w, _ = world2object(pts_box_in_o,
                                    None,
                                    pose[intersection_map],
                                    theta_y[intersection_map],
                                    dim[intersection_map],
                                    inverse=True)
        pts_box_in_w_new, _ = object2world(pts_box_in_o,
                                       None,
                                       pose[intersection_map],
                                       theta_y[intersection_map],
                                       dim[intersection_map])
        pts_box_in_w = torch.squeeze(pts_box_in_w)

        # Get all intersecting rays in unit length and the corresponding z_vals
        rays_o_in_w = rays_o[:, None, :].repeat(1, pose.shape[1], 1)[intersection_map]
        rays_d_in_w = rays_d[:, None, :].repeat(1, pose.shape[1], 1)[intersection_map]
        # Account for non-unit length rays direction
        z_vals_in_w = torch.norm(pts_box_in_w - rays_o_in_w, dim=1) / torch.norm(rays_d_in_w, dim=-1)

        if one_intersec_per_ray:
            # Get just nearest object point on a single ray
            z_vals_in_w, intersection_map, first_in_only = get_closest_intersections(z_vals_in_w,
                                                                                     intersection_map,
                                                                                     N_rays=rays_o.shape[0],
                                                                                     N_obj=theta_y.shape[1])
            if mask_only:
                return intersection_map
            # Get previous calculated values just for first intersections
            z_ray_in_o = z_ray_in_o[first_in_only]
            z_ray_out_o = z_ray_out_o[first_in_only]
            pts_box_in_o = pts_box_in_o[first_in_only]
            pts_box_in_w = pts_box_in_w[first_in_only]
            rays_o_in_w = rays_o_in_w[first_in_only]
            rays_d_in_w = rays_d_in_w[first_in_only]

        # Get the far intersection points and integration steps for each ray-box intersection in world and object frames
        pts_box_out_o = rays_o_o[intersection_map] + z_ray_out_o[:, None] * dirs_o[intersection_map]
        pts_box_out_w, _ = world2object(pts_box_out_o,
                                       None,
                                       pose[intersection_map],
                                       theta_y[intersection_map],
                                       dim[intersection_map],
                                       inverse=True)

        pts_box_out_w_new, _ = object2world(pts_box_out_o,
                                        None,
                                        pose[intersection_map],
                                        theta_y[intersection_map],
                                        dim[intersection_map],)
        pts_box_out_w = torch.squeeze(pts_box_out_w)
        z_vals_out_w = torch.norm(pts_box_out_w - rays_o_in_w, dim=1) / torch.norm(rays_d_in_w, dim=-1)

        # Get viewing directions for each ray-box intersection
        viewdirs_box_o = dirs_o[intersection_map]
        viewdirs_box_w = 1 / torch.norm(rays_d_in_w, dim=1)[:, None] * rays_d_in_w

    else:
        # In case no ray intersects with any object return empty lists
        if mask_only:
            return intersection_map
        z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        z_ray_out_o = z_ray_in_o = []
    return pts_box_in_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w, \
           pts_box_in_o, viewdirs_box_o, z_ray_in_o, z_ray_out_o, \
           intersection_map


def get_closest_intersections(z_vals_w, intersection_map, N_rays, N_obj):
    """Reduces intersections given by z_vals and intersection_map to the first intersection along each ray

    Args:
        z_vals_w: All integration steps for all ray-box intersections in world coordinates [n_intersections,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [n_intersections, 2]
        N_rays: Total number of rays
        N_obj: Total number of objects

    Returns:
        z_vals_w: Integration step for the first ray-box intersection per ray in world coordinates [N_rays,]
        intersection_map: Mapping from flat array to ray-box intersection matrix [N_rays, 2]
        id_first_intersect: Mapping from all intersection related values to first intersection only [N_rays,1]

    """
    # Flat to dense indices
    # Create matching ray-object intersectin matrix with index for all z_vals
    device = z_vals_w.device
    tmp_mask = intersection_map.view(-1)
    cumsum = torch.cumsum(tmp_mask, dim=0)
    id_z_vals = torch.where(tmp_mask, cumsum - 1, torch.zeros_like(tmp_mask, dtype=torch.long))
    id_z_vals = id_z_vals.view(N_rays, N_obj)

    # Create ray-index array
    id_ray = torch.arange(N_rays).long().to(device)

    # Flat to dense values
    # Scatter z_vals in world coordinates to ray-object intersection matrix
    
    ids = id_z_vals[intersection_map]
    # ids = torch.where(intersection_map, id_z_vals, torch.zeros_like(id_z_vals)).long()
    z_scattered_1d = torch.gather(z_vals_w.view(-1), 0, ids)
    z_scattered = torch.zeros(N_rays, N_obj, device=device, dtype=z_vals_w.dtype)
    z_scattered.masked_scatter_(intersection_map, z_scattered_1d.view(-1))

    # Set empty intersections to 1e10
    z_scattered_nz = torch.where(z_scattered == 0, torch.ones_like(z_scattered) * 1e10, z_scattered)

    # Get minimum values along each ray and corresponding ray-box intersection id
    id_min = torch.argmin(z_scattered_nz, dim=1)
    id_reduced = torch.cat((id_ray[:, None], id_min[:, None]), dim=1)
    z_vals_w_reduced = torch.gather(z_scattered, 1, id_reduced[:, 1][:, None])

    # Remove all rays w/o intersections (min(z_vals_reduced) == 0)
    id_non_zeros = torch.nonzero(z_vals_w_reduced != 0, as_tuple=True)[0]
    if len(id_non_zeros) != N_rays:
        z_vals_w_reduced = torch.gather(z_vals_w_reduced, 0, id_non_zeros.unsqueeze(1))
        id_reduced = torch.gather(id_reduced, 0, id_non_zeros.unsqueeze(1))

    # Get intersection map only for closest intersection to the ray origin
    # intersection_map_reduced_0 = torch.zeros(N_rays, N_obj, dtype=torch.bool, device=device)
    # for i in range(N_rays):
    #     intersection_map_reduced_0[id_reduced[i][0], id_reduced[i][1]] = True
    intersection_map_reduced = torch.zeros(N_rays, N_obj, dtype=torch.bool, device=device)
    intersection_map_reduced[id_reduced[:, 0], id_reduced[:, 1]] = True
    # id_first_intersect_0 = torch.zeros(N_rays, device=device)
    # for i in range(N_rays):
    #     id_first_intersect_0[i] = id_z_vals[id_reduced[i][0], id_reduced[i][1]]
    id_first_intersect = torch.zeros(N_rays, device=device)
    id_first_intersect = id_z_vals[id_reduced[:, 0], id_reduced[:, 1]]

    return z_vals_w_reduced, intersection_map_reduced, id_first_intersect


def combine_z(z_vals_bckg, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj, N_samples_obj):
    """Combines and sorts background node and all object node intersections along a ray

    Args:
        z_vals_bckg: integration step along each ray [N_rays, N_samples]
        z_vals_obj_w:  integration step of ray-box intersection in the world frame [n_intersects, N_samples_obj
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at ray-box intersection
        N_rays: Amount of rays
        N_samples: Amount of samples along each ray
        N_obj: Maximum number of objects
        N_samples_obj: Number of samples per object

    Returns:
        z_vals:  [N_rays, N_samples + N_samples_obj*N_obj, 4]
        id_z_vals_bckg: [N_rays, N_samples + N_samples_obj*N_obj] boolean
        id_z_vals_obj: [N_rays, N_samples + N_samples_obj*N_obj] boolean similar with intersection_map
    """
    device = intersection_map.device
    if z_vals_obj_w is None:
        z_vals_obj_w_sparse = torch.zeros(N_rays, N_obj * N_samples_obj, device=device, dtype=torch.float32)
    else:
        z_vals_obj_w_sparse = torch.zeros(N_rays, N_obj, N_samples_obj, device=device, dtype=torch.float32)
        z_vals_obj_w_sparse[intersection_map] = z_vals_obj_w
        z_vals_obj_w_sparse = z_vals_obj_w_sparse.reshape(N_rays, N_samples_obj * N_obj)

    sample_range = torch.arange(0, N_rays).to(device)
    obj_range = (sample_range[:, None, None]).repeat(1, N_obj, N_samples_obj)

    # Get ids to assign z_vals to each model
    tmp_bckg = None
    if z_vals_bckg is not None:
        id_z_vals_bckg = torch.zeros(N_rays, N_samples + N_samples_obj*N_obj, dtype=bool, device=device)
        id_z_vals_obj = torch.zeros(N_rays, N_samples + N_samples_obj*N_obj, dtype=bool, device=device)
        if len(z_vals_bckg.shape) < 2:
            z_vals_bckg = z_vals_bckg[:, None]
        # Combine and sort z_vals along each ray
        z_vals, _ = torch.sort(torch.cat((z_vals_obj_w_sparse, z_vals_bckg), dim=1), dim=1)

        tmp_bckg = torch.searchsorted(z_vals, z_vals_bckg.contiguous())

        # id_z_vals_bckg[torch.arange(N_rays)[:, None], tmp_bckg] = True

        bckg_range = sample_range[:, None, None].repeat(1, N_samples, 1)
        id_z_vals_bckg = torch.cat(
            (bckg_range, torch.searchsorted(z_vals, z_vals_bckg.contiguous())[..., None]), dim=-1
        )

    else:
        z_vals, _ = torch.sort(z_vals_obj_w_sparse, dim=1)
        id_z_vals_bckg = None
        id_z_vals_obj = torch.zeros(N_rays, N_samples_obj*N_obj, dtype=bool, device=device)

    tmp_obj = torch.searchsorted(z_vals, z_vals_obj_w_sparse.contiguous())

    # idx = torch.nonzero(tmp_obj)
    # src = tmp_obj[idx[:, 0]]
    
    # tmp0 = idx[:, :1]
    # tmp1 = torch.gather(src, -1, idx[:, 1:2])
    # id_z_vals_obj[tmp0, tmp1] = True

    id_z_vals_obj = torch.cat(
        [
            obj_range[..., None],
            torch.reshape(torch.searchsorted(z_vals, z_vals_obj_w_sparse), (N_rays, N_obj, N_samples_obj))[..., None]
        ],
        dim=-1
    )

    ##############
    # using either float32 or float64, there are always some similar z_vals_obj_w_sparse,
    # so that the sorted_id(tmp_obj) has some same id.
    # this will lead to true value of id_z_vals_obj lacks.
    # however, if judge during the loop, it is too slow.
    # still, out of loop, it is too slow.
    # add try except in run_nsg.py
    ##############

    # if int(torch.sum(id_z_vals_obj)) != len(z_vals_obj_w)*N_samples_obj:  # n_intersect*N_sample_obj
    #     for i, x in enumerate(tmp_obj):
    #         tmp_id = x!=0
    #         if int(torch.sum(tmp_id)) + 1 == len(torch.unique(x)):
    #             id_z_vals_obj[i, x[tmp_id]] = True
    #         else:
    #             tmp_id_first_index = {}
    #             duplicate_value = {}
    #             for j, id in enumerate(tmp_id):
    #                 if int(id.cpu()) not in tmp_id_first_index:
    #                     tmp_id_first_index[int(id.cpu())] = j
    #                 elif int(id.cpu()) not in duplicate_value:
    #                     duplicate_value[int(id.cpu())] = [j]
    #                 else:
    #                     duplicate_value[int(id.cpu())].append(j)
    #             for duplicate_place, duplicate_index in duplicate_value.items():
    #                 for j, duplicate_index_i in enumerate(duplicate_index):
    #                     tmp_id[duplicate_index_i] = duplicate_place + j + 1
    #             id_z_vals_obj[i, x[tmp_id]] = True

    return z_vals, id_z_vals_bckg, id_z_vals_obj, tmp_bckg, tmp_obj

    
