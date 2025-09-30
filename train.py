import torch
from tqdm import tqdm
from kitti_dataset import *
from run_nsg_helper import *
from manipulation import *
import datetime
from torch.utils.tensorboard import SummaryWriter

def run_network(step, inputs, viewdirs, fn, latents_all=None, patches=None, encoder=None):
    """Prepares inputs and applies network 'fn'.
    """
    device = inputs.device
    latents_flat = None
    input_obj_pose_flat = None
    inputs_flat = torch.reshape(inputs[..., :3], [-1, 3])  # rgb
    if inputs.shape[-1] > 3:
        # inputs: 3 + 2
        # 0, 1, ..., num_selected_frames, 
        # num_selected_frames + 1, ..., num_selected_frames * 2

        # frame_id + obj_id
        inputs_id = inputs[:, 0, 3:5] # [N, 2]

        if encoder is not None:
            unq, idx = torch.unique(inputs_id, dim=0, return_inverse=True)
            tuple_list = [(int(x[0].cpu()), int(x[1].cpu())) for x in unq]

            batch_patch = torch.cat([patches[x] for x in tuple_list])
            latents = encoder(batch_patch)

            color_lat = latents['color'][idx]
            density_lat = latents['density'][idx]

            latents_flat = {}
            latents_flat['color'] = color_lat[:, None, :].repeat(1, inputs.shape[1], 1)
            latents_flat['color'] = latents_flat['color'].reshape(-1, color_lat.shape[-1])
            latents_flat['density'] = density_lat[:, None, :].repeat(1, inputs.shape[1], 1)
            latents_flat['density'] = latents_flat['density'].reshape(-1, density_lat.shape[-1])
        elif latents_all.get('MultiHead', None) is not None:
            unq, idx = torch.unique(inputs_id, dim=0, return_inverse=True)
            tuple_list = [(int(x[0].cpu()), int(x[1].cpu())) for x in unq]
            if latents_all['MultiHead']:
                color_lat_unq = torch.stack([latents_all[x]['color'] for x in tuple_list]).to(device)
                density_lat_unq = torch.stack([latents_all[x]['density'] for x in tuple_list]).to(device)

                color_lat = color_lat_unq[idx]
                density_lat = density_lat_unq[idx]

                latents_flat = {}
                latents_flat['color'] = color_lat.repeat(1, inputs.shape[1], 1)
                latents_flat['color'] = latents_flat['color'].reshape(-1, color_lat.shape[-1])
                latents_flat['density'] = density_lat.repeat(1, inputs.shape[1], 1)
                latents_flat['density'] = latents_flat['density'].reshape(-1, density_lat.shape[-1])
            else:
                lat_unq = torch.stack([latents_all[x] for x in tuple_list]).to(device)
                lat = lat_unq[idx]
                latents_flat = lat.repeat(1, inputs.shape[1], 1)
                latents_flat = latents_flat.reshape(-1, lat.shape[-1])
        else:
            lat = torch.stack([latents_all[int(x.cpu())] for x in inputs_id[:, 1]]).to(device)
            latents_flat = lat.repeat(1, inputs.shape[1], 1)
            latents_flat = latents_flat.reshape(-1, lat.shape[-1])
     
    if viewdirs is not None:
        input_dirs = viewdirs[:, None, :3].expand(inputs[..., :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        if viewdirs.shape[-1] > 3:

            input_obj_pose = viewdirs[:, None, 3:].expand(
                inputs[..., :3].shape[0], inputs[..., :3].shape[1], 3)
            input_obj_pose_flat = torch.reshape(input_obj_pose, [-1, 3])
    
    outputs_flat, latents_shape = fn(step, inputs_flat, input_dirs_flat, latents_flat, input_obj_pose_flat)

    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs, latents_shape

def create_nerf(args, num_progressive_split, scene_bounds, pose_scale_factors, device):
    skips = [4]
    
    kwargs_all = []
    for i in range(num_progressive_split):
        model = {}
        bckg_model = NeRF(
            D=args.netdepth, W=args.netwidth, D_views=args.netdepth//2, 
            skips=skips, latent_sz=args.latent_size, n_freq_posenc=args.multires, 
            n_freq_posenc_views=args.multires_views, n_freq_posenc_obj=args.multires_obj, 
            use_original_latents=args.use_original_latents
        ).to(device)
        grad_vars = list(bckg_model.parameters())
        model['model_bckg'] = bckg_model

        if not args.bckg_only:
            for obj_class in args.scene_classes:
                model_name = 'model_class_'+str(int(obj_class)).zfill(4)
                if args.use_autorf_decoder:
                    model_obj = ConditionalRenderer(
                        use_embed_viewdirs=False, use_obj_pose=args.use_objpose
                    ).to(device)
                else:
                    model_obj = NeRF(
                        D=args.netdepth, W=args.netwidth, D_views=args.netdepth//2, 
                        skips=skips, latent_sz=args.latent_size, n_freq_posenc=args.multires, 
                        n_freq_posenc_views=args.multires_views, n_freq_posenc_obj=args.multires_obj, 
                        use_original_latents=args.use_original_latents
                    ).to(device)
                grad_vars += list(model_obj.parameters())
                model[model_name] = model_obj
            
            if not args.opt_encoder:
                encoder = None
            elif args.use_autorf_decoder:
                encoder = MultiHeadImageEncoder().to(device)
                grad_vars += list(encoder.parameters())
            elif not args.use_original_latents:
                encoder = LatentCodeEncoder().to(device)
                grad_vars += list(encoder.parameters())
            else:
                encoder = None
        else:
            encoder = None
        
        proposal_param_dict = {}
        if args.sampling_method == 'proposal':
            proposal_mlps = []
            for j in range(args.num_proposal_iterations):
                proposal_mlps.append(ProposalMLP(D=args.num_layer_proposal, W=args.hidden_dim_proposal).to(device))
            proposal_param_dict['mlps'] = proposal_mlps
            proposal_param_dict['N_samples_proposal'] = [int(x) for x in args.num_proposal_samples_per_ray.split(',')]
            proposal_param_dict['N_samples'] = int(args.num_nerf_samples_per_ray*args.expand_bound_factor)

        network_query_fn = \
        lambda step, inputs, viewdirs, network_fn, latents_all, patches, encoder: \
        run_network(step, inputs, viewdirs, network_fn, latents_all, patches, encoder)
    
        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        # optimizer = torch.optim.RAdam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        render_kwargs_train = {
            'network_query_fn' : network_query_fn,
            'networks_dict' : model,
            'optimizer': optimizer,
            'encoder': encoder,
            'latent_vector_dict': None,
            'N_samples' : int(args.N_samples*args.expand_bound_factor),
            'N_samples_obj' : args.N_samples_obj,
            'N_obj': args.max_input_objects if not args.bckg_only else 0,
            'obj_only': args.obj_only,
            'bckg_only': args.bckg_only,
            'pose_scale_factor': pose_scale_factors[i],
            'scene_objects': args.scene_objects,
            'perturb' : args.perturb,
            'white_bkgd' : args.white_bkgd,
            'raw_noise_std' : args.raw_noise_std,
            'sampling_method': args.sampling_method,
            'proposal_params': proposal_param_dict,
            'is_train': True,
            'use_rays_d': args.use_old_raw2outputs
        }
        render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.
        render_kwargs_test['is_train'] = False
        
        if args.render_only:
            kwargs_all.append(render_kwargs_test)
        else:
            kwargs_all.append(render_kwargs_train)

    start = 0
    start_epoch = 0
    basedir = args.basedir
    expname = args.expname

    ########## Load Ckpt #########
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 
                 'tar' in f]
    
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Loading ckpt from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step'] + 1
        start_epoch = int(ckpt_path.split('/')[-1].split('_')[0][-3:])

        for i in range(num_progressive_split):
            print('Loading progressive split', i)
            if not args.render_only:
                kwargs_all[i]['optimizer'].load_state_dict(ckpt[i]['optimizer_state_dict'])

            kwargs_all[i]['networks_dict']['model_bckg'].load_state_dict(ckpt[i]['network_state_dict'])
            print('Reloaded bckg model')

            if not args.bckg_only:
                if kwargs_all[i]['encoder'] is not None:
                    kwargs_all[i]['encoder'].load_state_dict(ckpt[i]['encoder_state_dict'])
                    print('Reloaded encoder')
                for k in ckpt[i]:
                    if 'class' in k:
                        kwargs_all[i]['networks_dict'][k].load_state_dict(ckpt[i][k])
                        print('Reloaded dynamic model {}'.format(k))
            
            if args.sampling_method == 'proposal':
                for j in range(args.num_proposal_iterations):
                    kwargs_all[i]['proposal_params']['mlps'][j].load_state_dict(ckpt[i][f'proposal_mlp_{j}'])
                    print(f'Reloaded proposal mlp {j}')
            if ckpt[i]['latents_all'] is not None:
                kwargs_all[i]['latent_vector_dict'] = ckpt[i]['latents_all']
                print('Reloaded latents_all')

    if kwargs_all[0]['latent_vector_dict'] is None:
        if args.use_original_latents:
            latents_all = get_latents_all_nsg(args.latent_size, args.scene_objects, device)
            print('Initialized all latents randomly')
        else:
            if args.use_autorf_decoder:
                lat_encoder = MultiHeadImageEncoder().to(device)
                if args.pretrain_encoder_path is not None:
                    encoder_ckpt = torch.load(args.pretrain_encoder_path, map_location=device)
                    local_dict = {}
                    for k, v in encoder_ckpt.items():
                        name_split = k.split('.')[2:]
                        name = ''
                        for n in name_split:
                            name += n
                            name += '.'
                        name = name[:-1]
                        local_dict[name] = v
                    lat_encoder.load_state_dict(local_dict)
                    print('Reload encoder params from autorf pretrained model.')
            else:
                lat_encoder = LatentCodeEncoder().to(device)
            latents_all = get_latents_all(args.datadir, lat_encoder, device)
            print('Initialized all latents from img')
        for i in range(num_progressive_split):
            kwargs_all[i]['latent_vector_dict'] = latents_all
    
    return kwargs_all, start, start_epoch


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, use_rays_d=False):

    if raw_noise_std > 0.:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std
    else:
        noise = 0.
    
    if use_rays_d:
        # TODO 
        # this way of calculating weights may be more robust, 
        # it is ok to cat large number at the end of delta
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples]
        dists = torch.cat([dists, 1e7 * torch.ones_like(z_vals[..., :1])], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        disp_map = 1. / torch.maximum(1e-7 * torch.ones_like(depth_map), depth_map / acc_map)


        deltas = dists

    else:
        rgb = torch.sigmoid(raw[..., :3])

        weights = get_weights_from_density(z_vals, raw[..., 3] + noise)
        
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
        acc_map = torch.sum(weights, dim=-1)
        
        depth_map = torch.sum(weights * z_vals, dim=-1) / (acc_map + 1e-7)
        disp_map = 1 / depth_map

        deltas = z_vals[..., 1:] - z_vals[..., :-1]

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, deltas


def render(rays, obj, gt, chunk=1024*32, **kwargs):

    if kwargs['is_train']:
        # rays.shape [N_rays, 11]
        # obj.shape [N_rays, max_obj*11] -> [N_rays, max_obj, 11]
        obj = obj.reshape(obj.shape[0], -1, 11)
        # gt['rgb'].shape [N_rays, 3]
        # gt['depth'].shape [N_rays]

        rays_on_obj, rays_to_remove = get_all_ray_3dbox_intersection(rays, obj)

        if kwargs['bckg_only']:
            print('Removing objects from scene.')
            rays = rays[~rays_on_obj]
            obj = None
            print(rays.shape)
        elif kwargs['obj_only']:
            print('Extracting objects from background.')
            rays_bckg = None
            rays = rays[rays_on_obj]
            print(rays.shape)
        else:
            rays_bckg = rays[~rays_on_obj]
            rays = rays[rays_on_obj]

        if not kwargs['bckg_only']:
            rays, obj, gt, obj_mask_idx = resample_rays(rays, rays_bckg, obj, gt, rays_on_obj, \
                                            torch.where(rays_on_obj)[0], kwargs['scene_objects'], kwargs['objects_meta'])
    else:
        if kwargs['bckg_only']:
            obj = None
        else:
            # rays.shape [H*W, 11]
            # obj.shape [max_obj, 11] -> [H*W, max_obj, 11]
            obj = obj[None, ...].repeat(rays.shape[0], 1, 1)


    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(
            ray_batch=rays[i:i+chunk],
            obj_batch=obj[i:i+chunk] if obj is not None else None,
            **kwargs
        )
        for k in ret:
            if isinstance(ret[k], dict):  # latent_shape
                if k not in all_ret:
                    all_ret[k] = {}

                for model_name in ret[k]:  # or obj_name
                    if model_name not in all_ret[k]:
                        all_ret[k][model_name] = ret[k][model_name]
                    else:
                        all_ret[k][model_name] = torch.cat((all_ret[k][model_name], ret[k][model_name]))
            elif isinstance(ret[k], list):
                if k not in all_ret:
                    all_ret[k] = ret[k]
                else:
                    for i in range(len(ret[k])):
                        all_ret[k][i] = torch.cat((all_ret[k][i], ret[k][i]))
            else:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

    for k in all_ret:
        if isinstance(all_ret[k], dict): continue
        if 'list' in k or 'loss' in k: continue
        try:
            all_ret[k] = torch.cat(all_ret[k])
        except RuntimeError as e:
            tmp_ret = []
            for tmp in all_ret[k]:
                if tmp.shape[-1] != 0:
                    tmp_ret.append(tmp)
            all_ret[k] = torch.cat(tmp_ret)
    if kwargs['is_train'] and not kwargs['bckg_only']:
        all_ret['obj_mask_idx'] = obj_mask_idx
    k_extract = ['rgb_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return [gt] + ret_list + [ret_dict]

@torch.no_grad()
def render_path(rays, obj, pose, depth_gt, H, W, K, chunk, savedir=None, saveidx=None, obj_manipulation=None, cam_manipulation=None, **kwargs):
    if obj_manipulation is not None:
        obj = manipulation_obj_pose(obj, obj_manipulation, **kwargs)
    if cam_manipulation is not None:
        pose = manipulation_cam_pose(pose, cam_manipulation, **kwargs)
        new_rays_od = torch.stack([get_rays(H, W, K, pose_i) for pose_i in pose])  # [n_pose, 2, H, W, 3]
        new_rays_o = new_rays_od[:, 0, ...].flatten(1, 2)  # [n_pose, H, W, 3] --> [n_pose, H*W, 3]
        new_rays_d = new_rays_od[:, 1, ...].flatten(1, 2)
        rays = rays.repeat(len(pose), 1, 1)
        rays[:, :, :3] = new_rays_o
        rays[:, :, 3:6] = new_rays_d
    
    ret = []
    ret_metrics = []
    
    # rays.shape [num_to_render, H*W, 11], obj.shape [num_to_render, H*W, 11]
    for rays_idx, render_rays_i in enumerate(rays):
        for obj_idx, render_obj_i in enumerate(obj):
            if obj_manipulation is not None or cam_manipulation is not None:
                save_with_gt = False
                ret_metrics = [-1, -1, -1, -1]
            else:
                save_with_gt = True
                rgb_gt = render_rays_i[:, 6:9].clone().reshape(H, W, 3)
            
            viewdirs = render_rays_i[:, 3:6]
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            render_rays_i[:, 6:9] = viewdirs
            
            _, rgb_pred, _, extras = render(rays=render_rays_i, obj=render_obj_i, gt=None, chunk=chunk, **kwargs)
            rgb_pred = torch.clamp(rgb_pred, min=0, max=1).reshape(H, W, 3)
            depth_pred_np = extras['depth_map'].reshape(H, W).cpu().numpy()

            ret.append(rgb_pred)
            ret.append(depth_pred_np)

            depth_pred_np *= kwargs['pose_scale_factor']
            
            if save_with_gt and (savedir is None or saveidx is None):
                psnr, ssim, lpips_vgg, lpips_alex = metrics(rgb_pred, rgb_gt, None)
                ret_metrics = [psnr, ssim, lpips_vgg, lpips_alex]

                continue

            savename = str(saveidx).zfill(3) + '_' + str(rays_idx) + '_' + str(obj_idx)

            if save_with_gt:
                metrics_savefile = os.path.join(savedir, 'metrics.txt')
                psnr, ssim, lpips_vgg, lpips_alex = metrics(rgb_pred, rgb_gt, metrics_savefile)
                ret_metrics = [psnr, ssim, lpips_vgg, lpips_alex]

                if kwargs['obj_only']:
                    rgb_gt[~kwargs['mask']] = 0
                    rgb_pred[~kwargs['mask']] = 0
                rgb_obj_gt = rgb_gt[kwargs['mask']]
                rgb_obj_pred = rgb_pred[kwargs['mask']]
                print('obj_psnr: ', mse2psnr(img2mse(rgb_obj_gt, rgb_obj_pred)).item())
                
                filename = os.path.join(savedir, savename+'_'+str(int(psnr*100))+'.png')
                img2save = np.vstack((to8b(rgb_gt.cpu().numpy()), to8b(rgb_pred.cpu().numpy())))
                imageio.imwrite(filename, img2save)

                depth_gt_np = depth_gt[0].cpu().numpy()
                depth_gt_mask = depth_gt_np != 0

                fig, axs = plt.subplots(2, 2)
                fig.tight_layout()
                axs[0, 0].set_title('depth_pred, vmax=depth_pred.max()')
                axs[0, 0].axis('off')
                axs[0, 0].imshow(depth_pred_np, cmap='plasma', vmin=0, vmax=depth_pred_np.max())
                depth_pred_np[~depth_gt_mask] = 0.0
                depth_residual = np.abs(depth_gt_np - depth_pred_np)
                depth_avg_residual = depth_residual.sum()/depth_gt_mask.sum()
                axs[0, 1].set_title('depth_gt')
                axs[0, 1].axis('off')
                axs[0, 1].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=depth_gt_np.max())
                axs[1, 0].set_title('depth_pred')
                axs[1, 0].axis('off')
                axs[1, 0].imshow(depth_pred_np, cmap='plasma', vmin=0, vmax=depth_gt_np.max())
                axs[1, 1].set_title('depth_residual, avg={:3f}'.format(depth_avg_residual))
                axs[1, 1].axis('off')
                axs[1, 1].imshow(depth_residual, cmap='plasma', vmin=0, vmax=depth_gt_np.max())

                filename = os.path.join(savedir, savename+'_depth.png')
                plt.savefig(filename)
                plt.close()
            else:
                filename = os.path.join(savedir, savename+'.png')
                img2save = to8b(rgb_pred.cpu().numpy())
                imageio.imwrite(filename, img2save)

                plt.figure()
                plt.axis('off')
                plt.imshow(depth_pred_np, cmap='plasma', vmin=0, vmax=depth_pred_np.max())
                plt.title('depth_pred')

                filename = os.path.join(savedir, savename+'_depth.png')
                plt.savefig(filename)
                plt.close()

    return ret, ret_metrics


def render_rays(ray_batch, network_query_fn, networks_dict, N_samples, N_samples_obj, 
                obj_batch=None, 
                obj_only=False, 
                N_obj=0,
                encoder=None,
                patches=None,
                latent_vector_dict=None,
                perturb=0,
                white_bkgd=False,
                raw_noise_std=0,
                sampling_method=None,
                proposal_params=None,
                **kwargs):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, 6:9]
    near, far = ray_batch[:, 9:10], ray_batch[:, 10:]
    device = ray_batch.device
    latent_shape = None
    truncation_mask = None

    weights_list = []
    z_vals_list = []

    if not obj_only:
        # sample
        # For training object models only sampling close to the objects is performed
        if (sampling_method == 'planes' or sampling_method == 'planes_plus') and kwargs['plane_bds'] is not None:
            # Sample at ray plane intersection (Neural Scene Graphs)
            pts, z_vals = plane_pts([rays_o, rays_d], [kwargs['plane_bds'], kwargs['plane_normal'], kwargs['plane_delta']], \
                                    kwargs['id_planes'], near, method=sampling_method)
            N_importance = 0
        elif sampling_method == 'proposal':
            proposal_mlps = proposal_params['mlps']
            num_proposal_network_iterations = len(proposal_mlps)
            num_proposal_samples_per_ray = proposal_params['N_samples_proposal']
            num_nerf_samples_per_ray = proposal_params['N_samples']
            weights = None
            for i_level in range(num_proposal_network_iterations + 1):
                is_prop = i_level < num_proposal_network_iterations
                num_samples = num_proposal_samples_per_ray[i_level] if is_prop else num_nerf_samples_per_ray
                # z_vals in Euclidean space, s_dists in s-space
                if i_level == 0:
                    z_vals, s_dists, spacing_to_euclidean_fn = UniformLinDispPiecewiseSample(num_samples, near, far, perturb)
                else:
                    if kwargs['is_train']:
                        weights = torch.pow(weights, kwargs['anneal'])
                    z_vals, s_dists = PDFSample(num_samples, s_dists, spacing_to_euclidean_fn, weights, perturb)  # [N_rays, N_samples + 1]
                if is_prop:
                    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
                    density = proposal_mlps[i_level](pts)
                    weights = get_weights_from_density(z_vals, density)
                    weights_list.append(weights)
                    z_vals_list.append(z_vals)
                else:
                    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[:, 1:, None]
                    z_vals = z_vals[:, 1:]
                    N_samples = num_samples
        else:
            # Sample along ray (vanilla NeRF)
            pts, z_vals = sample_along_ray([rays_o, rays_d], near, far, N_samples, N_rays, sampling_method, perturb)

    if obj_batch is None:
        # bckg only
        pts = pts.to(torch.float32)
        viewdirs = viewdirs.to(torch.float32)
        raw, _ = network_query_fn(kwargs['step'], pts, viewdirs, networks_dict['model_bckg'], None, None, None)
    else:
        pts_box_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
        pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
        intersection_map = box_pts(
            [rays_o, rays_d], obj_batch[..., 7:10], obj_batch[..., 10], dim=obj_batch[..., 4:7],
            one_intersec_per_ray=False)
        
        if z_vals_in_o is None or len(z_vals_in_o) == 0:
            if obj_only:
                # No computation necessary if rays are not intersecting with any objects and no background is selected

                rgb_map = torch.zeros(N_rays, 3, dtype=torch.float32, device=device)
                acc_map = torch.ones(N_rays, dtype=torch.float32, device=device)
                weights = torch.zeros(N_rays, N_obj*N_samples_obj, dtype=torch.float32, device=device)
                depth_map = torch.zeros(N_rays, dtype=torch.float32, device=device)
                deltas = torch.zeros(N_rays, N_obj*N_samples_obj-1, dtype=torch.float32, device=device)

                ret = {'rgb_map': rgb_map, 'acc_map': acc_map,
                        'weights': weights, 'depth_map': depth_map, 'deltas': deltas}

                return ret
            else:
                # TODO: Do not return anything for no intersections.
                z_vals_obj_w = torch.zeros(1, dtype=torch.float32, device=device)
                intersection_map = torch.zeros(N_rays, N_obj, dtype=bool, device=device)
        else:
            if not obj_only:
                truncation_mask = (z_vals[torch.nonzero(intersection_map)[:, 0]] > z_vals_in_w[:, None]) \
                    & (z_vals[torch.nonzero(intersection_map)[:, 0]] < z_vals_out_w[:, None])

            n_intersect = len(z_vals_in_o)

            obj_intersect = obj_batch[intersection_map]
            obj_intersect = obj_intersect[:, None, :].repeat(1, N_samples_obj, 1)
            
            if N_samples_obj > 1:
                z_vals_box_o = torch.linspace(0., 1., N_samples_obj)[None, :].repeat(n_intersect, 1).to(device) * \
                                (z_vals_out_o - z_vals_in_o)[:, None]
            else:
                z_vals_box_o = torch.tensor(1/2)[None, None].repeat(n_intersect, 1).to(device) * \
                            (z_vals_out_o - z_vals_in_o)[:, None]      
            pts_box_samples_o = pts_box_o[:, None, :] + viewdirs_box_o[:, None, :] * z_vals_box_o[..., None]
            obj_intersect_transform = obj_intersect.reshape(-1, obj_intersect.shape[-1])
            pts_box_samples_w, _ = world2object(pts_box_samples_o.reshape(-1, 3), None,
                                                obj_intersect_transform[..., 7:10],
                                                obj_intersect_transform[..., 10],
                                                dim=obj_intersect_transform[..., 4:7],
                                                inverse=True)
            pts_box_samples_w = pts_box_samples_w.reshape(n_intersect, N_samples_obj, 3)
            rays_o_intersected = rays_o[:, None, :].repeat(1, N_obj, 1)[intersection_map]  # (n_intersect, 3)
            rays_o_intersected = rays_o_intersected[:, None, :].repeat(1, N_samples_obj, 1)  # (n_intersect, N_samples_obj, 3)
            z_vals_obj_w = torch.norm(pts_box_samples_w - rays_o_intersected, dim=-1).to(torch.float32)

            # Extract objects
            obj_ids = obj_intersect[..., 2]  # [n_intersect, N_samples_obj]
            object_unique, object_index = torch.unique(obj_ids.flatten(), return_inverse=True) # unique_id, index
            # Extract classes
            obj_class = obj_intersect[..., 3]
            class_unique, class_index = torch.unique(obj_class.flatten(), return_inverse=True)

            inputs = pts_box_samples_o  # xyz

            viewdirs_obj = torch.cat([viewdirs_box_o, obj_intersect[:, 0, 7:10]], dim=1)
        
        if not obj_only:
            z_vals, id_z_vals_bckg, id_z_vals_obj, tmp_bckg, tmp_obj = combine_z(z_vals,
                                                                z_vals_obj_w if z_vals_in_o is not None else None,
                                                                intersection_map,
                                                                N_rays,
                                                                N_samples,
                                                                N_obj,
                                                                N_samples_obj)
        else:
            z_vals, _, id_z_vals_obj, tmp_bckg, tmp_obj = combine_z(None, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj,
                                                 N_samples_obj)

        if not obj_only:
            # run bckg model
            raw = torch.zeros(N_rays, N_samples + N_obj*N_samples_obj, 4, device=device)
            pts = pts.to(torch.float32)
            viewdirs = viewdirs.to(torch.float32)
            raw_bckg, _ = network_query_fn(kwargs['step'], pts, viewdirs, networks_dict['model_bckg'], None, None, None)
            if truncation_mask is not None:
                raw_bckg[torch.nonzero(intersection_map)[:, 0]][truncation_mask][:, 3] = 0.0
            raw[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1]] += raw_bckg
            
        else:
            raw = torch.zeros(N_rays, N_obj*N_samples_obj, 4, device=device)

        if z_vals_in_o is not None and len(z_vals_in_o) != 0:
            # Loop over classes c and evaluate each models f_c for all latent object describtor

            for c, class_id in enumerate(class_unique):
                # Ignore background class
                if class_id >= 0:
                    input_indices = class_index==c
                    input_indices = input_indices.reshape(-1, inputs.shape[1])  # (n_intersect, N_samples_obj)
                    model_name = 'model_class_' + str(int(np.array(class_id.cpu()))).zfill(4)

                    assert model_name in networks_dict

                    obj_network_fn = networks_dict[model_name]
                    frame_ids = obj_intersect[:, :, 0]
                    inputs = torch.cat([inputs, frame_ids.unsqueeze(-1), obj_ids.unsqueeze(-1)], dim=-1)
                    inputs_obj_c = inputs[input_indices].reshape(-1, inputs.shape[1], inputs.shape[2])
                    viewdirs_obj_c = viewdirs_obj[input_indices[:, :6]].reshape(-1, 6)

                    # Predict RGB and density from object model
                    raw_k, tmp_latent_shape = network_query_fn(kwargs['step'], inputs_obj_c, viewdirs_obj_c, obj_network_fn, latent_vector_dict, patches, encoder)
                    
                    if tmp_latent_shape is not None:
                        # store latent_shape(appearance) for each obj
                        # tmp_latent_shape length: number of class_index==c
                        for o, object_id in enumerate(object_unique):
                            obj_latent_indices = object_index==o
                            obj_latent_indices = obj_latent_indices.reshape(-1, inputs.shape[1])  # (n_intersect, N_samples_obj)
                            # obj_latent_indices is now the indices for all classes, larger than len(tmp_latent_shape)
                            # and obj_latent_indices[input_indices] is the indices for current class, same shape as len(tmp_latent_shape)
                            obj_name = 'latent_obj_' + str(int(np.array(object_id.cpu()))).zfill(4)
                            obj_latent = tmp_latent_shape[obj_latent_indices[input_indices]]
                            obj_latent = obj_latent.reshape(-1, N_samples_obj, obj_latent.shape[-1])
                            obj_latent = obj_latent[:, 0, :].reshape(-1, obj_latent.shape[-1])
                            if obj_latent is None or len(obj_latent) == 0:
                                continue
                            if latent_shape is None:
                                latent_shape = {}
                                latent_shape[obj_name] = obj_latent
                            elif latent_shape.get(obj_name) is None:
                                latent_shape[obj_name] = obj_latent
                            else:
                                latent_shape[obj_name] = torch.cat(latent_shape[obj_name], obj_latent)  # dict (n, 256)
                    
                    obj_index = id_z_vals_obj[intersection_map][input_indices]
                    raw[obj_index[:, 0], obj_index[:, 1]] += raw_k.reshape(-1, raw_k.shape[-1])
    
    rgb_map, disp_map, acc_map, weights, depth_map, deltas = \
        raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, use_rays_d=kwargs['use_rays_d'])
    weights_list.append(weights)
    z_vals_list.append(torch.cat([z_vals[:, :1], z_vals], dim=-1))

    ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map,
           'weights': weights, 'deltas': deltas}
    if latent_shape is not None:
        ret['latent_shape'] = latent_shape
    ret['weights_list'] = weights_list
    ret['z_vals_list'] = z_vals_list

    for k in ret:
        if isinstance(ret[k], dict) or isinstance(ret[k], list): continue
        if torch.isnan(ret[k]).any():
            print(f'Tensor ret[{k}] contains NaN values.')
        if torch.isinf(ret[k]).any() and kwargs['is_train']:
            print(f'Tensor ret[{k}] contains Inf values.')
    
    return ret

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='the specified ckpt path to load')
    parser.add_argument("--sampling_method", type=str, default=None,
                        help='method to sample points along the ray options: None / lindisp / squaredist / plane')
    parser.add_argument("--bckg_only", action='store_true',
                        help='removes rays associated with objects from the training set to train just the background model.')
    parser.add_argument("--obj_only", action='store_true',
                        help='Train object models on rays close to the objects only.')
    parser.add_argument("--use_inst_segm", action='store_true',
                        help='Use an instance segmentation map to select a subset from all sampled rays')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_samples_obj", type=int, default=3,
                        help='number of samples per ray and object')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_shadows", action='store_true',
                        help='use pose of an object to predict shadow opacity')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_obj", type=int, default=4,
                        help='log2 of max freq for positional encoding (3D object location + heading)')
    parser.add_argument("--total_step", type=int, default=50000,
                        help='total reg iter of embedding frequency mask')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--remove_frame", type=int, default=-1,
                        help="Remove the ith frame from the training set")
    parser.add_argument("--remove_obj", type=int, default=None,
                        help="Option to remove all pixels of an object from the training")

    # render flags
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--manipulate", type=str, default=None,
                        help='Renderonly manipulation argument')

    # vkitti/kitti flags
    parser.add_argument("--dataset_type", type=str, default='kitti')
    parser.add_argument("--first_frame", type=str, default=0,
                        help='specifies the beginning of a sequence if not the complete scene is taken as Input')
    parser.add_argument("--last_frame", type=str, default=None,
                        help='specifies the end of a sequence')
    parser.add_argument("--max_input_objects", type=int, default=20,
                        help='Max number of object poses considered by the network, will be set automatically')
    parser.add_argument("--scene_objects", type=list,
                        help='List of all objects in the trained sequence')
    parser.add_argument("--scene_classes", type=list,
                        help='List of all unique classes in the trained sequence')
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Maximum scale for boxes to include shadows")
    parser.add_argument("--plane_type", type=str, default='uniform',
                        help='specifies how the planes are sampled')
    parser.add_argument("--near_plane", type=float, default=0.5,
                        help='specifies the distance from the last pose to the far plane')
    parser.add_argument("--far_plane", type=float, default=150.,
                        help='specifies the distance from the last pose to the far plane')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='num of iterations of weight ckpt saving')
    parser.add_argument("--i_render", type=int, default=100000)

    # train flags
    parser.add_argument("--use_original_latents", action='store_true',
                        help='whether to use original nsg latents')
    parser.add_argument("--use_autorf_decoder", action='store_true',
                        help='whether to use autorf conditional renderer for obj')
    parser.add_argument("--use_objpose", action='store_true',
                        help='whether to use objpose in conditional render')
    parser.add_argument("--opt_encoder", action='store_true')
    parser.add_argument("--use_old_raw2outputs", action='store_true')

    # latent settings
    parser.add_argument("--latent_size", type=int, default=128,
                        help='Size of the latent vector representing each of object of a class. If 0 no latent vector '
                             'is applied and a single representation per object is used.')
    parser.add_argument("--latent_balance", type=float, default=0.01,
                        help="Balance between image loss and latent loss")
    parser.add_argument("--weight_latent_loss", type=float, default=1.0)

    parser.add_argument("--weight_render_loss", type=float, default=0,
                        help="Weight of render loss, which is mse loss of rendered sam result and original sam result")
    
    # data sample
    parser.add_argument("--train_every", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=-1)
    parser.add_argument("--side", type=str, default='both',
                        help='both, left, or right')

    # dataset and preprocess
    parser.add_argument("--load_imagefilenames", action='store_true', 
                        help='load image filenames or images when initializing dataset')
    parser.add_argument("--use_collider", action='store_true',
                        help='whether to use near_far_collider')
    parser.add_argument("--progressive_param_t", type=float, default=30.0)
    parser.add_argument("--progressive_param_angle", type=float, default=30.0)
    parser.add_argument("--progressive_param_min_num", type=int, default=10)
    parser.add_argument("--pretrain_encoder_path", type=str, default=None)
    parser.add_argument("--bound_setting", type=int, default=0,
                        help='0 is suds style, 1 is cube')
    parser.add_argument("--expand_bound_factor", type=float, default=1.0)

    # depth loss
    parser.add_argument("--weight_depth_loss", type=float, default=0)
    parser.add_argument("--weight_sigma_loss", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=1.0)

    # proposal
    parser.add_argument("--num_proposal_iterations", type=int, default=0)
    parser.add_argument("--hidden_dim_proposal", type=int, default=16)
    parser.add_argument("--num_layer_proposal", type=int, default=1)
    parser.add_argument("--num_proposal_samples_per_ray", type=str, default='16,32')
    parser.add_argument("--num_nerf_samples_per_ray", type=int, default=8)
    parser.add_argument("--proposal_weights_anneal_slope", type=float, default=10.0)
    parser.add_argument("--proposal_weights_anneal_max_num_iters", type=int, default=1000)
    parser.add_argument("--weight_distortion_loss", type=float, default=0.002)
    parser.add_argument("--weight_interlevel_loss", type=float, default=1.0)

    ##########Tensorf-base config##########
    parser.add_argument("--update_AlphaMask_list", type=int, default=[3000, 6000, 9000], nargs='+')
    parser.add_argument("--refinement_speedup_factor", type=float, default=1.0, 
                        help="Divides number of iterations in scheduling. Does not apply to progressive optimization.")
    parser.add_argument(
        "--shadingMode", type=str, default="MLP_Fea_late_view", help="which shading mode to use"
    )
    parser.add_argument("--pos_pe", type=int, default=0, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=0, help="number of pe for view")
    parser.add_argument(
        "--fea_pe", type=int, default=0, help="number of pe for features"
    )
    parser.add_argument(
        "--featureC", type=int, default=128, help="hidden feature channel in MLP"
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument("--N_voxel_init", type=int, default=64**3)
    parser.add_argument("--N_voxel_final", type=int, default=640**3)
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-5,
        help="shift density in softplus; making density = 0  when feature == 0",
    )
    parser.add_argument("--n_lamb_sigma", type=int, default=[8, 8, 8], action="append")
    parser.add_argument("--n_lamb_sh", type=int, default=[24, 24, 24], action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.001,
        help="mask points in ray marching",
    )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.0001,
        help="threshold for creating alpha mask volume",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--lr_upsample_reset", type=bool, default=True)

    return parser

def train(args, device):

    if args.obj_only and args.bckg_only:
        print('Object and background can not set as train only at the same time.')
        return

    starts = args.first_frame.split(',')
    ends = args.last_frame.split(',')
    if len(starts) != len(ends):
        print('Number of sequences is not defined. Using the first sequence')
        args.first_frame = int(starts[0])
        args.last_frame = int(ends[0])
    else:
        args.first_frame = [int(val) for val in starts]
        args.last_frame = [int(val) for val in ends]
    selected_frames = [args.first_frame, args.last_frame]
    progressive_param = {
        't': args.progressive_param_t,
        'angle': args.progressive_param_angle,
        'min_num': args.progressive_param_min_num
    }
    data_sample_param = {
        'train_every': args.train_every,
        'test_every': args.test_every,
        'is_train': False if args.render_only else True,
        'render_test': args.render_test,
        'side': args.side
    }
    kitti_data = kitti_tracking_dataset(
        args.datadir,
        selected_frames,
        device,
        args.near_plane,
        args.far_plane,
        args.box_scale,
        progressive_param=progressive_param,
        data_sample_param=data_sample_param,
        use_collider=args.use_collider,
        N_samples_plane=args.N_samples if args.sampling_method=='planes' else 0,
        plane_type=args.plane_type if args.sampling_method=='planes' else None,
        load_image=not args.load_imagefilenames,
        N_rand=args.N_rand,
        bound_setting=args.bound_setting,
        expand_bound_factor=args.expand_bound_factor
    )
    print('Initialized kitti tracking dataset')

    args.scene_objects, args.scene_classes = [], []
    for v in kitti_data.objects_meta.values():
        if v[0] not in args.scene_objects:
            args.scene_objects.append(v[0])
        if v[4] not in args.scene_classes:
            args.scene_classes.append(v[4])
    args.max_input_objects = kitti_data.visible_objects.shape[1]

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    kwargs_all, start, start_epoch = create_nerf(
        args, 
        len(kitti_data.split), 
        kitti_data.scene_bounds, 
        kitti_data.pose_scale_factors, 
        device
    )

    global_step = start

    for i, x in enumerate(kwargs_all):
        x['objects_meta'] = kitti_data.objects_meta
        if len(kitti_data.plane_bds_dicts) > 0:
            x.update(kitti_data.plane_bds_dicts[i])

    # Create log dir and copy the config file
    if args.render_only:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                    'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        f = os.path.join(testsavedir, 'args.txt')
    else:
        f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        if args.render_only:
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                    'test' if args.render_test else 'path', start))
            f = os.path.join(testsavedir, 'config.txt')
        else:
            f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.render_only:
        print('RENDER ONLY')
        H, W, K = kitti_data.H, kitti_data.W, kitti_data.calib.P2[:3, :3]
        kitti_data.is_train = False
        render_dataloader = torch.utils.data.DataLoader(kitti_data)

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
        if args.obj_only:
            testsavedir = testsavedir + '_obj'
        elif args.bckg_only:
            testsavedir = testsavedir + '_bckg'
        
        if args.manipulate is None:
            obj_manipulation = None
            cam_manipulation = None
        elif 'obj' in args.manipulate:
            testsavedir = testsavedir + '_' + args.manipulate
            obj_manipulation = args.manipulate.split('obj_')[-1]
            cam_manipulation = None
        elif 'cam' in args.manipulate:
            testsavedir = testsavedir + '_' + args.manipulate
            obj_manipulation = None
            cam_manipulation = args.manipulate.split('cam_')[-1]

        os.makedirs(testsavedir, exist_ok=True)
        print('Start rendering for', len(render_dataloader), 'images at', testsavedir)

        pbar_render = tqdm(range(len(render_dataloader)), desc='render: ')
        for saveidx, (rays, obj, split_id, mask, depth, patch, pose, patch_mask) in enumerate(render_dataloader):
            # rays.shape [1, H*W, 11], obj.shape [1, max_obj, 11], split_id.shape [1]
            # depth.shape [1, H, W], patch: p.shape [1, 3, patch_H, patch_W]=[1, 3, 80, 120], pose.shape [1, 4, 4]
            kwargs = kwargs_all[split_id]
            kwargs['patches'] = patch
            kwargs['mask'] = mask.squeeze().to(bool)  # [H, W] only matters when render obj_only
            kwargs['patch_masks'] = patch_mask  # [1, h, w, 3]
            kwargs['step'] = global_step
            if obj_manipulation is not None:
                kwargs['visible_objects'] = torch.tensor(kitti_data.visible_objects, dtype=torch.float32, device=device)
            _, [psnr, ssim, lpips_vgg, lpips_alex] = \
                render_path(rays, obj, pose, depth, H, W, K, args.chunk, testsavedir, saveidx, \
                            obj_manipulation, cam_manipulation, **kwargs)
            pbar_render.set_description('render {:03d}: psnr: {:.5f}, ssim: {:.5f}, lpips(vgg): {:.5f}, lpips(alex): {:.5f}' \
                                    .format(saveidx, psnr, ssim, lpips_vgg, lpips_alex))
            pbar_render.update(1)
        pbar_render.close()

        print('Done. Saved at', testsavedir)

        return

    train_dataloader = torch.utils.data.DataLoader(kitti_data)
    
    N_iter = 1000001
    i_epoch = start_epoch
    
    print('start training')
    curr_time = datetime.datetime.now()
    timestamp = datetime.date.strftime(curr_time, '%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter('logs/'+args.expname+'/'+timestamp)
    pbar = tqdm(range(start+1, N_iter))
    i = start + 1
    while i < N_iter:
        for rays_rgb_obj, split_id, _, depth, patch, _ in train_dataloader:
            # rays_rgb_obj.shape [1, N_rand, 11+max_obj*11]:
            #   rays_o*3, rays_d*3, rgb*3, near, far
            # obj = rays_rgb_obj[:, :, 11:]
            #   frame_id, cam_id, obj_id, obj_type, dim*3, xyz*3, yaw
            # split_id.shape [1, N_rand]
            # depth.shape [1, N_rand]
            # patch: p.shape [1, 3, patch_H, patch_W]=[1, 3, 80, 120]
            
            rays_rgb_obj = rays_rgb_obj.squeeze()

            rgb_gt = rays_rgb_obj[:, 6:9].clone()  # [N_rand, 3]
            viewdirs = rays_rgb_obj[:, 3:6]
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            rays_rgb_obj[:, 6:9] = viewdirs
            depth_gt = depth.squeeze()  # [N_rand]

            gs_loss = []
            gs_psnr = []

            train_frac = np.clip((i-1) / args.proposal_weights_anneal_max_num_iters, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, args.proposal_weights_anneal_slope)

            split_id = split_id.squeeze()
            unq_id, unq_idx = torch.unique(split_id, return_inverse=True)
            for j in range(len(unq_id)):
                current_split = unq_id[j]
                kwargs_all[current_split]['patches'] = patch
                kwargs_all[current_split]['anneal'] = anneal
                kwargs_all[current_split]['step'] = i
                gt = {}
                gt['rgb'] = rgb_gt[unq_idx==j, :]
                gt['depth'] = depth_gt[unq_idx==j]
                
                gt_resampled, rgb, _, extras = render(
                    rays_rgb_obj[unq_idx==j, :11],
                    rays_rgb_obj[unq_idx==j, 11:],
                    gt, chunk=args.chunk, **kwargs_all[current_split]
                )

                kwargs_all[current_split]['optimizer'].zero_grad()

                rgb_gt_resampled = gt_resampled['rgb'] 
                depth_gt_resampled = gt_resampled['depth']

                img_loss = img2mse(rgb, rgb_gt_resampled)
                loss = img_loss
                psnr = mse2psnr(img_loss)

                save_prefix = 'split_'+str(current_split.item())
                tensorboard_save_dict = {}
                tensorboard_save_dict['img_loss'] = img_loss.item()
                tensorboard_save_dict['psnr'] = psnr.item()

                # Compute psnr on obj pixels
                if 'obj_mask_idx' in extras.keys():
                    obj_mask_idx = extras['obj_mask_idx']
                    obj_psnr = mse2psnr(img2mse(rgb[obj_mask_idx], rgb_gt_resampled[obj_mask_idx]))
                    del extras['obj_mask_idx']
                    tensorboard_save_dict['obj_psnr'] = obj_psnr.item()
                else:
                    tensorboard_save_dict['obj_psnr'] = 0

                # Add loss for depth
                if args.weight_depth_loss > 0:
                    depth_mask = depth_gt_resampled > 0
                    pose_scale_factor = kwargs_all[current_split]['pose_scale_factor']
                    # depth_loss = img2mse(extras['depth_map'][depth_mask]*pose_scale_factor, depth_gt_resampled[depth_mask])
                    depth_loss = img2mse(extras['depth_map']*pose_scale_factor*depth_mask, depth_gt_resampled*depth_mask) \
                                    / (torch.sum(depth_mask) + 1e-8)
                    weights = extras['weights']
                    z_vals = extras['z_vals_list'][-1]
                    if args.use_old_raw2outputs:
                        deltas = extras['deltas']
                    else:
                        # TODO
                        # deltas = torch.cat([extras['deltas'], 1e7 * torch.ones_like(z_vals[:, :1])], dim=-1)
                        deltas = torch.cat([torch.zeros_like(z_vals[:, :1]), extras['deltas']], dim=-1)
                    sigma_loss = - torch.log(weights + 1e-7) \
                                 * torch.exp(-(z_vals[:, 1:]*pose_scale_factor - depth_gt_resampled[:, None]) ** 2 / (2 * args.sigma)) \
                                 * deltas*pose_scale_factor * depth_mask[:, None]
                    sigma_loss = torch.sum(sigma_loss, dim=1).mean()
                    depth_loss *= args.weight_depth_loss
                    sigma_loss *= args.weight_sigma_loss
                    loss += depth_loss
                    loss += sigma_loss
                    tensorboard_save_dict['depth_loss'] = depth_loss.item()
                    tensorboard_save_dict['sigma_loss'] = sigma_loss.item()

                # Add loss for latent code
                if args.weight_latent_loss > 0:
                    if extras.get('latent_shape', None) is None:
                        tensorboard_save_dict['latent_loss'] = 0
                    else:
                        latent_loss = 0
                        if args.use_original_latents:
                            # original nsg latent loss trick
                            args.weight_latent_loss = args.latent_balance
                            for latent_i in extras['latent_shape'].values():
                                latent_loss += torch.norm(latent_i, dim=0).sum()
                        else:
                            for latent_i in extras['latent_shape'].values():
                                # latent_i: (n, latent_size)
                                if latent_i.shape[0] == 1:
                                    latent_loss += torch.norm(latent_i)
                                latent_variance = torch.var(latent_i, dim=0)  # (1, latent_size)
                                latent_loss += torch.norm(latent_variance)
                        latent_loss *= args.weight_latent_loss
                        tensorboard_save_dict['latent_loss'] = latent_loss.item()
                        loss += latent_loss
                
                # Add loss for proposal sample
                if args.weight_distortion_loss > 0:
                    distort_loss = args.weight_distortion_loss * distortion_loss(extras['weights_list'], extras['z_vals_list'])
                    loss += distort_loss
                    tensorboard_save_dict['distortion_loss'] = distort_loss.item()
                if args.weight_interlevel_loss > 0:
                    interlvl_loss = args.weight_interlevel_loss * interlevel_loss(extras['weights_list'], extras['z_vals_list'])
                    loss += interlvl_loss
                    tensorboard_save_dict['interlevel_loss'] = interlvl_loss.item()

                tensorboard_save_dict['loss'] = loss.item()
                writer.add_scalars(save_prefix, tensorboard_save_dict, i)
                gs_loss.append(loss.item())
                gs_psnr.append(psnr.item())

                loss.backward()
                kwargs_all[current_split]['optimizer'].step()

                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                for param_group in kwargs_all[current_split]['optimizer'].param_groups:
                    param_group['lr'] = new_lrate

            gs_loss = np.array(gs_loss)
            gs_psnr = np.array(gs_psnr)
            pbar.set_description('E={:04d}, split={:01d}, global_step={:06d}, loss={:5f}, psnr={:.5f}' \
                                .format(i_epoch, current_split, global_step, gs_loss.mean(), gs_psnr.mean()))
            pbar.update(1)
            i += 1
            global_step += 1

            if i % args.i_img == 0 and i > 0:
                H, W, K = kitti_data.H, kitti_data.W, kitti_data.calib.P2[:3, :3]
                kitti_data.is_train = False
                render_idx = np.random.randint(len(kitti_data))
                rays, obj, split_id, _, depth, patch, pose, _ = kitti_data.__getitem__(render_idx)
                # rays.shape [H*W, 11], obj.shape [max_obj, 11], type(split_id) int, 
                # depth.shape [H, W], patch: p.shape [3, patch_H, patch_W]=[3, 80, 120], pose.shape [4, 4]
                kwargs = kwargs_all[split_id].copy()
                kwargs['patches'] = {k: v.unsqueeze(0) for k, v in patch.items()}
                kwargs['perturb'] = False
                kwargs['raw_noise_std'] = 0.
                kwargs['is_train'] = False
                rgb_gt = rays[:, 6:9].clone().reshape(H, W, 3)
                [rgb, depth_np], [psnr, ssim, lpips_vgg, lpips_alex] = \
                    render_path(rays[None, ...], obj[None, ...], pose[None, ...], depth[None, ...], H, W, K, args.chunk, **kwargs)
                depth_colored = apply_colormap(depth_np)
                writer.add_image('render_image', torch.cat((rgb_gt, rgb)), i, dataformats='HWC')
                writer.add_image('render_depth', depth_colored, i, dataformats='HWC')
                writer.add_scalar('render_psnr', psnr, i)
                writer.add_scalar('render_ssim', ssim, i)
                writer.add_scalar('render_lpips(vgg)', lpips_vgg, i)
                writer.add_scalar('render_lpips(alex)', lpips_alex, i)
                kitti_data.is_train = True

            if i % args.i_weights == 0 and i > 0:
                path = os.path.join(basedir, expname, 'epoch{:03d}_iter{:06d}.tar'.format(i_epoch, i))
                savedict = {
                    'global_step': global_step
                }
                for j in range(len(kwargs_all)):
                    savedict_j = {
                        'optimizer_state_dict': kwargs_all[j]['optimizer'].state_dict(),
                        'network_state_dict': kwargs_all[j]['networks_dict']['model_bckg'].state_dict(),
                        'latents_all': kwargs_all[j]['latent_vector_dict']
                    }
                    if kwargs_all[j]['encoder'] is not None:
                        savedict_j['encoder_state_dict'] = kwargs_all[j]['encoder'].state_dict()
                    for k in kwargs_all[j]['networks_dict'].keys():
                        if 'class' in k:
                            savedict_j[k] = kwargs_all[j]['networks_dict'][k].state_dict()
                    if args.sampling_method == 'proposal':
                        for m in range(args.num_proposal_iterations):
                            savedict_j[f'proposal_mlp_{m}'] = kwargs_all[j]['proposal_params']['mlps'][m].state_dict()
                    savedict[j] = savedict_j
                
                torch.save(savedict, path)
                print('Saved checkpoints at', path)

            if i % args.i_render == 0 and i > 0:
                H, W, K = kitti_data.H, kitti_data.W, kitti_data.calib.P2[:3, :3]
                kitti_data.is_train = False
                render_dataloader = torch.utils.data.DataLoader(kitti_data)

                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                        'test' if args.render_test else 'path', i))

                os.makedirs(testsavedir, exist_ok=True)
                print('Start rendering for', len(render_dataloader), 'images at', testsavedir)

                pbar_render = tqdm(range(len(render_dataloader)), desc='render: ')
                for saveidx, (rays, obj, split_id, _, depth, patch, pose, _) in enumerate(render_dataloader):
                    # rays.shape [1, H*W, 11], obj.shape [1, max_obj, 11], split_id.shape [1]
                    # depth.shape [1, H, W], patch: p.shape [1, 3, patch_H, patch_W]=[1, 3, 80, 120], pose.shape [1, 4, 4]
                    kwargs = kwargs_all[split_id].copy()
                    kwargs['patches'] = patch
                    kwargs['perturb'] = False
                    kwargs['raw_noise_std'] = 0.
                    kwargs['is_train'] = False
                    _, [psnr, ssim, lpips_vgg, lpips_alex] = render_path(rays, obj, pose, depth, H, W, K, args.chunk, testsavedir, saveidx, **kwargs)
                    pbar_render.set_description('render {:03d}: psnr: {:.5f}, ssim: {:.5f}, lpips(vgg): {:.5f}, lpips(alex): {:.5f}' \
                                            .format(saveidx, psnr, ssim, lpips_vgg, lpips_alex))
                    pbar_render.update(1)
                pbar_render.close()

                print('Done. Saved at', testsavedir)
                kitti_data.is_train = True
        
        # shuffle data after an epoch
        i_epoch += 1
        kitti_data.shuffle()
    
    writer.close()
    pbar.close()

if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    parser = config_parser()
    args = parser.parse_args()
    train(args, device)
