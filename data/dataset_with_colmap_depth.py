# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, re
import warnings
import numpy as np
import cv2
from scipy.interpolate import griddata
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data_depth, batch_parse_llff_poses


class DatasetWithColmapDepth(Dataset):
    #def __init__(self, args, mode, name, scenes=None, **kwargs):
    def __init__(self, root_dir, split, n_views=3, levels=1, downSample=1.0, max_len=-1):
        base_dir = os.path.join(root_dir, 'data/')

        dataset_list = ['dtu', 'ibrnet_collected', 'real_iconic_noface']  #
        #self.name = name
        
        self.testskip = 4
        scenes = None
        warnings.filterwarnings('ignore')

        self.mode = split  # train / test / validation
        self.num_source_views = n_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        self.scene_path = []

        self.scale_colmap = []
        self.name = []
        #scenes = os.listdir(base_dir)
        list_prefix='new'
        count = 0
        for name in dataset_list:
            #if (split == "val" or split == "test") and scenes !=None:
            #    scenes = scenes
            #else:
            scenes = []
            if split == "train":
                file_list = os.path.join(base_dir, name, list_prefix+"_train.lst")
            elif split == "val":
                file_list = os.path.join(base_dir, name, list_prefix+"_val.lst")
            elif split == "test":
                file_list = os.path.join(base_dir, name, list_prefix+"_test.lst")

            with open(file_list, "r") as f:
                scenes = [x.strip() for x in f.readlines()]
            
            for i, scene in enumerate(scenes):
                scene_path = os.path.join(base_dir, name, scene)

                _, poses, bds, render_poses, i_test, rgb_files, sc = load_llff_data_depth(scene_path, load_imgs=False, factor=1)
                near_depth = np.min(bds)
                far_depth = np.max(bds)
                intrinsics, c2w_mats = batch_parse_llff_poses(poses)

                if split == 'train':
                    i_train = np.array(np.arange(int(poses.shape[0])))
                    i_render = i_train
                else:# name=='nerf_llff_data' or name=='nerf_synthetic':
                    if name == 'dtu':
                        #i_train = np.array([0, 8, 13, 22,25,28, 40, 43, 48])  ## pixel-nerf
                        i_train = np.array([25, 21, 33, 22, 14, 15, 26, 30, 31, 35, 34, 43, 46, 29, 16, 36])
                        i_test = np.array([32, 24, 23, 44])
                        #i_test = np.array([j for j in np.arange(int(poses.shape[0])) if (j not in i_train)])[::self.testskip]#

                    else:
                        i_test = np.arange(poses.shape[0])[::self.testskip]
                        i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                            (j not in i_test and j not in i_test)])
                    i_render = i_test
                
                self.train_intrinsics.append(intrinsics[i_train])
                self.train_poses.append(c2w_mats[i_train])
                self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
                num_render = len(i_render)
                self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
                self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
                self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
                self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
                self.render_train_set_ids.extend([i+count]*num_render)
                self.scale_colmap.extend([sc]*num_render)
                self.scene_path.extend([scene_path]*num_render)
                self.name.extend([name]*num_render)
            count += len(scenes)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.

        #depth, colmap_depth, colmap_mask = self.load_depth(rgb_file)

        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]

        depth_range = self.render_depth_range[idx]
        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # For dtu foreground mask.
        """
        if self.name == 'dtu':
            mask_foreground = self.read_pfm(rgb_file, img_size)            
        else:
            mask_foreground = None
        """

        if self.mode == 'train':
            id_render = train_rgb_files.index(rgb_file)
            num_select = self.num_source_views #+ np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            num_select = self.num_source_views
        subsample_factor = 1
        num_select_inner = min(num_select, train_poses.shape[0] - 1)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                num_select_inner,
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)


        assert id_render not in nearest_pose_ids, rgb_file
        
        src_rgbs = []
        src_cameras = []
        #src_depths = []
        #src_colmap_depths = []
        #src_colmap_masks = []

        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.

            #src_depth, src_colmap_depth, src_colmap_mask = self.load_depth(train_rgb_files[id])

            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)
            #src_depths.append(src_depth)
            #src_colmap_depths.append(src_colmap_depth)
            #src_colmap_masks.append(src_colmap_mask)
    
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        #src_depths = torch.stack(src_depths)
        #src_colmap_depths = torch.stack(src_colmap_depths)
        #src_colmap_masks = torch.stack(src_colmap_masks)

        # align scale depth
        #sc = self.scale_colmap[idx]
        
        #depth = self.align_scales(depth[None], colmap_depth[None], colmap_mask[None], sc=sc)
        #src_depths = self.align_scales(src_depths, src_colmap_depths, src_colmap_masks, sc=sc).unsqueeze(1)

        # random flip.
        if self.mode == 'train' and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)
            #depth = torch.flip(depth, dims=[-1])
            #src_depths = torch.flip(src_depths, dims=[-1])

        # np.array to torch.tensor.
        rgb = torch.from_numpy(rgb[..., :3])
        camera = torch.from_numpy(camera)
        src_rgbs = torch.from_numpy(src_rgbs[..., :3])
        src_cameras = torch.from_numpy(src_cameras)
        

        # warp source images to target image.
        #depths_warped = self.warp_src_to_tgt(camera, src_rgbs, src_depths, src_cameras)
        
        # depth range.
        if self.name[idx] in ['dtu', 'nerf_synthetic']:
            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.1])
        elif self.name[idx] in ['nerf_llff_data', 'ibrnet_collected', 'scannet', 'real_iconic_noface', 'RealEstate10K-subset']:
            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])
        else:
            import pdb;pdb.set_trace()

        # Convert output spec to mvsnerf.
        imgs = torch.cat([src_rgbs, rgb[None]], dim=0).permute(0, 3, 1, 2)
        depths_h = torch.zeros(self.num_source_views+1, 1, 1)
        
        c2w_t = camera[18:].reshape(4, 4)
        c2w_s = src_cameras[:, 18:].reshape(-1, 4, 4)
        c2ws = torch.cat([c2w_s, c2w_t[None]], dim=0)
        w2cs = torch.inverse(c2ws)

        near_fars = depth_range[None].repeat(self.num_source_views+1, 1)
        
        K_t = camera[2:18].reshape(-1, 4, 4)#[:, :3, :3]
        K_s = src_cameras[:, 2:18].reshape(-1, 4, 4)#[:, :3, :3]
        intrinsics = torch.cat([K_s, K_t], dim=0)#[:, :3, :3]
        
        proj_mats_ls = intrinsics @ w2cs
        affine_mat = proj_mats_ls.clone()
        affine_mat_inv = torch.inverse(affine_mat)

        proj_mats = []
        for i in range(self.num_source_views+1):
            if i==0:
                ref_proj_inv = torch.inverse(proj_mats_ls[i])
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mats_ls[i] @ ref_proj_inv]
        proj_mats = torch.stack(proj_mats)[:, :3]
        intrinsics = intrinsics[:, :3, :3]

        return {'images': imgs,
                'depths_h': depths_h,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'near_fars': near_fars,
                'proj_mats': proj_mats,
                'intrinsics': intrinsics,
                #'view_ids': 
                'affine_mat': affine_mat,
                'affine_mat_inv': affine_mat_inv,
        }

        """
        return {'rgb': rgb,
                'camera': camera,
                'rgb_path': rgb_file,
                'src_rgbs': src_rgbs,
                'src_cameras': src_cameras,
                'depth_range': depth_range,
                #'depth': depth_warped_median,
                #'warped_depths': depths_warped,
                #'src_depths': src_depths,
                #'depth_consistency': depth_consistency,
                #'mask_foreground': mask_foreground,
                }
        """
    def load_depth(self, rgb_file):
        
        depth_file = rgb_file.replace("/images/", "/depth_dense/")+'.pt'
        colmap_file = rgb_file.replace("/images/", "/depth/")+'.pt'
        depth = torch.load(depth_file)
        colmap = torch.load(colmap_file)
        
        colmap_depth = colmap[0]
        colmap_mask = colmap[1]#.bool()
        assert colmap_depth.shape[0] == 480

        H = colmap_depth.shape[-2]
        W = colmap_depth.shape[-1]
        depth = torch.nn.functional.interpolate(depth[None, None], size=[H, W] , mode='bilinear', align_corners=True)[0, 0]
        #colmap_depth = torch.nn.functional.interpolate(colmap_depth[None, None], size=[H, W] , mode='nearest')[0, 0]
        #colmap_mask = torch.nn.functional.interpolate(colmap_mask[None, None], size=[H, W], mode='nearest')[0, 0]

        colmap_mask = colmap_mask.bool()
        return depth, colmap_depth, colmap_mask

    def read_pfm(self, rgb_file, img_size):
        # rect_001_3_r5000 -> depth_map_0000
        imgname = rgb_file.split('/')[-1]
        filedir = '/'.join(rgb_file.split('/')[:-1])
        imgid = int(imgname.split('_')[1]) -1
        filename = os.path.join(filedir, "depth_map_{0:0=4d}.pfm".format(imgid)).replace('/images/', '/mask_mvsnerf/')
        file = open(filename, 'rb')
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        file.close()

        mask_foreground = torch.from_numpy(data.copy()).bool().float()
        mask_foreground = F.interpolate(mask_foreground[None, None], size=img_size, mode='nearest',)
        mask_foreground = mask_foreground.squeeze()

        #return data, scale
        return mask_foreground

    def align_scales(self, depth_priors, colmap_depths, colmap_masks, sc):
        ratio_priors = []
        for i in range(depth_priors.shape[0]):
            ratio_priors.append(np.median(colmap_depths[i][colmap_masks[i]]) / np.median(depth_priors[i][colmap_masks[i]]))
        ratio_priors = np.stack(ratio_priors)
        ratio_priors = ratio_priors[:, np.newaxis, np.newaxis]

        depth_priors = depth_priors * sc * ratio_priors #align scales
        return depth_priors

    def resize_images(self, images, intrinsics, h_resize=480):
        H = images.shape[-3]
        W = images.shape[-2]

        ratio = W / H
        
        w_resize = int(ratio*h_resize)
        resize_factor = h_resize / H
        intrinsics_resized = intrinsics.copy()
        intrinsics_resized[:2, :3] *= resize_factor
        images_resized = torch.from_numpy(images)
        images_resized = F.interpolate(images_resized.unsqueeze(0).permute(0, 3, 1, 2), size=[h_resize, w_resize]).permute(0, 2, 3, 1)[0]
        images_resized = images_resized.numpy()

        return images_resized, intrinsics_resized

    def warp_src_to_tgt(self, camera, src_rgbs, src_depths, src_cameras):
        n_views, height, width, _ = src_rgbs.shape

        K_t = camera[2:18].reshape(4,4)[:3, :3]
        c2w_t = camera[18:].reshape(4, 4)

        K_src = src_cameras[:, 2:18].reshape(-1, 4, 4)[:, :3, :3]
        c2w_src = src_cameras[:, 18:].reshape(-1, 4, 4)
        
        depths_warped = []

        # Batched warping
        x_pixel = torch.arange(width).reshape(1, 1,width).repeat([n_views,height,1])
        y_pixel = torch.arange(height).reshape(1, height,1).repeat([n_views,1,width])
        ones = torch.ones_like(y_pixel).reshape(n_views, -1)

        xy = torch.stack([x_pixel.reshape(n_views, -1), y_pixel.reshape(n_views, -1), ones], dim=1).float()
        xy = torch.bmm(torch.inverse(K_src), xy)

        xyz = src_depths.reshape(n_views, 1, -1).repeat(1, 3, 1) * xy
        xyz_i = torch.cat([xyz, ones[:, None].float()], dim=1)

        xyz_world = torch.bmm(c2w_src, xyz_i)
        xyz_t = torch.bmm(torch.inverse(c2w_t)[None].repeat(n_views, 1, 1), xyz_world)
        xyz_t = torch.bmm(K_t[None].repeat(n_views,1,1), xyz_t[:, :3, :])

        z_t = xyz_t[:, 2, :]
        xy_t = xyz_t[:, :2, :] / z_t[:, None]

        mask = (xy_t[:, 0] >= -0.5) & (xy_t[:, 0] < width-0.5) & (xy_t[:, 1] >= -0.5) & (xy_t[:, 1] < height-0.5)

        view_id = torch.arange(n_views).reshape(n_views, 1 ,1).repeat(1, 1, width*height)
        ixy_t = torch.cat([view_id, xy_t], dim=1)

        view_id = ixy_t[:, 0][mask]
        x = ixy_t[:, 1][mask]
        y = ixy_t[:, 2][mask]
        z = z_t[mask]        
        #rgb = src_rgbs.reshape(n_views, -1, 3)[mask]

        z, indices = torch.sort(z, descending=False)
        view_id = view_id[indices]
        x = x[indices]
        y = y[indices]
        #rgb = rgb[indices]

        depths = torch.zeros(n_views, height, width).float()
        view_id = view_id.long()
        x = x.long()
        y = y.long()
        # unique.
        pixel_id = view_id*width*height + y*width + x

        indices = np.argsort(pixel_id, kind='stable')
        pixel_id = pixel_id[indices]
        z = z[indices]        
        #rgb = rgb[indices]

        pixel_id, counts = torch.unique_consecutive(pixel_id, return_counts=True)
        counts_cumsum = torch.cumsum(counts, dim=0)
        indices = torch.roll(counts_cumsum, shifts=1)
        indices[0] = 0
        z = z[indices]
        #rgb = rgb[indices]
        depths = depths.reshape(-1)

        depths[pixel_id] = z
        depths = depths.reshape(n_views, 1, height, width)

        #rgbs = torch.zeros(n_views, height, width, 3).float().reshape(-1, 3)
        #rgbs[pixel_id] = rgb
        #rgbs = rgbs.reshape(n_views, height, width, 3)

        """
        for i in range(n_views):
            depth_vis = depths[i, 0] / torch.max(depths[i, 0])
            depths_vis = depth_vis.cpu()
            print(i, torch.sum(torch.ones_like(depths[i][depths[i]>0])))
            #imageio.imwrite('./warping_test/depth_{}_a.png'.format(i), depths_vis)
            imageio.imwrite('./warping_test/depth_{}_rgb.png'.format(i), rgbs[i])
        """
        #import pdb; pdb.set_trace()
        
        return depths

if __name__ == '__main__':
    from dotmap import DotMap
    args = DotMap()
    args.num_source_views = 10
    args.llffhold = 8
    #args.rootdir = "/mnt/disk4/dongwool/datasets/ibrnet_NerfingMVS"
    args.rootdir = "/mnt/disk4/dongwool/datasets/ibrnet_NerfingMVS"
    dataset = DatasetWithColmapDepth(args, mode='train',
                                #name='nerf_llff_data',
                                #name='dtu',
                                name='nerf_synthetic',
                                #name='ibrnet_collected_1',
                                scenes=['lego'],
                                )

    for data in dataset:
        print('hello')
        #print(data["support_images"].shape, data["support_depths"].shape, data["support_masks"].shape)
        #raise NotImplementedError

