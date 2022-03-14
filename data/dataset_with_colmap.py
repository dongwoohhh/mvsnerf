import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import imageio

from .ray_utils import *
from .data_utils import random_crop, random_flip, get_nearest_pose_ids

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg



def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    # 草，这个地方源代码没有乘这个blender2opencv，做这个操作相当于把相机转换到另一个坐标系了，和一般的nerf坐标系不同
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    #print('center in center_poses',poses_centered[:, :3, 3].mean(0))

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)z


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class DatasetWithColmap(Dataset):
    def __init__(self, root_dir, split='train', n_views=3, levels=1,  img_wh=None, downSample=1.0, max_len=-1, spheric_poses=True, load_ref=False):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        #self.args = args
        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views

        dataset_list = ['ibrnet_collected_1']#, 'ibrnet_collected_1', 'ibrnet_collected_2', 'real_iconic_noface']

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.testskip = 4


        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        list_prefix='new'
        count = 0
        
        for name in dataset_list:
            scenes = []
            if split == "train":
                file_list = os.path.join(self.root_dir, name, list_prefix+"_train.lst")
            elif split == "val":
                file_list = os.path.join(self.root_dir, name, list_prefix+"_val.lst")
            elif split == "test":
                file_list = os.path.join(self.root_dirdir, name, list_prefix+"_test.lst")
        
            with open(file_list, "r") as f:
                scenes = [x.strip() for x in f.readlines()]

            if name == 'dtu':
                factor=1.0
            elif name in ['ibrnet_collected_1']:
                factor=1.125
            elif name in ['nerf_llff_data', 'real_iconic_noface', 'ibrnet_collected_2']:
                factor=4.5
            else:
                factor=1.0

            for i, scene in enumerate(scenes):
                intrinsics, c2w_mats, bounds, rgb_files = self.read_meta(name, scene, factor)
                
                near_depth = np.min(bounds)*0.8
                far_depth = np.max(bounds)*1.2

                if split == 'train':
                    i_train = np.array(np.arange(int(c2w_mats.shape[0])))
                    i_render = i_train
                else:# name=='nerf_llff_data' or name=='nerf_synthetic':
                    if name == 'dtu':
                        #i_train = np.array([0, 8, 13, 22,25,28, 40, 43, 48])  ## pixel-nerf
                        i_train = np.array([25, 21, 33, 22, 14, 15, 26, 30, 31, 35, 34, 43, 46, 29, 16, 36])
                        i_test = np.array([32, 24, 23, 44])
                        #i_test = np.array([j for j in np.arange(int(poses.shape[0])) if (j not in i_train)])[::self.testskip]#

                    else:
                        i_test = np.arange(c2w_mats.shape[0])[::self.testskip]
                        i_train = np.array([j for j in np.arange(int(c2w_mats.shape[0])) if
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
            count += len(scenes)

        self.spheric_poses = spheric_poses
        self.define_transforms()

        self.white_back = False

    def read_meta(self, name, scene, factor):
        poses_bounds = np.load(os.path.join(self.root_dir, name, scene, 'poses_bounds.npy'))  # (N_images, 17)
        
        if factor == 1.0:
            image_paths = sorted(glob.glob(os.path.join(self.root_dir, name, scene , 'images/*')))
        else:
            image_paths = sorted(glob.glob(os.path.join(self.root_dir, name, scene , 'images_{}/*'.format(factor))))
        # load full resolution image then resize

        if self.split in ['train', 'val']:
            #print(len(poses_bounds) , len(image_paths),self.root_dir)
            assert len(poses_bounds) == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        #focal = [focal / factor, self.focal / factor]
        
        #intrinsic = np.array([[focal/factor, 0, W/factor/2., 0],
        #                      [0, focal/factor, H/factor/2., 0],
        #                      [0, 0, 1, 0],
        #                      [0, 0, 0, 1]])
        intrinsic = np.array([[focal/factor, 0, W/factor/2.],
                              [0, focal/factor, H/factor/2.],
                              [0, 0, 1]])

        intrinsics = np.repeat(intrinsic[None],len(poses_bounds), axis=0)
        intrinsics = torch.from_numpy(intrinsics).float()
        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        poses, pose_avg = center_poses(poses, self.blender2opencv)
        # print('pose_avg in read_meta', self.pose_avg)
        # self.poses = poses @ self.blender2opencv

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        #print('scale_factor', scale_factor)
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        return intrinsics, poses, bounds, image_paths, 


        # sub select training views from pairing file
        if os.path.exists('configs/pairs.th'):
            name = os.path.basename(self.root_dir)
            self.img_idx = torch.load('configs/pairs.th')[f'{name}_{self.split}']

            print(f'===> {self.split}ing index: {self.img_idx}')

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)


        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in self.img_idx:

            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)


            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
                # near plane is always at 1.0
                # near and far in NDC are always 0 and 1
                # See https://github.com/bmild/nerf/issues/34
            else:
                # near = self.bounds.min()
                # far = min(8 * near, self.bounds.max())  # focus on central object only
                near = self.bounds[i][0]*0.8
                far = self.bounds[i][1]*1.2  # focus on central object only

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         near * torch.ones_like(rays_o[:, :1]),
                                         far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)

        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
        elif 'val' == self.split:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        
        render_pose = self.render_poses[idx]
        render_intrinsic = self.render_intrinsics[idx]

        depth_range = self.render_depth_range[idx]
        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        #img_size = rgb.shape[1:]
        h, w = rgb.shape[1:]

        if self.split == 'train':
            id_render = train_rgb_files.index(rgb_file)
            num_select = self.n_views #+ np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            num_select = self.n_views
        
        subsample_factor = 1
        num_select_inner = min(num_select, train_poses.shape[0] - 1)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                num_select_inner,
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]

        for i, id in enumerate(nearest_pose_ids):
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            imgs.append(self.transform(src_rgb))


            c2w = torch.eye(4).float()
            c2w[:3] = torch.FloatTensor(train_poses[id])
            w2c = torch.inverse(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = torch.eye(4)
            intrinsic = train_intrinsics[id]
            intrinsics.append(intrinsic.clone())

            intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]

            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_l)
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

        # Cat target view.
        imgs.append(self.transform(rgb))
        c2w = torch.eye(4).float()
        c2w[:3] = torch.FloatTensor(render_pose)
        w2c = torch.inverse(c2w)
        c2ws.append(c2w)
        w2cs.append(w2c)

        proj_mat_l = torch.eye(4)
        intrinsic = render_intrinsic
        intrinsics.append(intrinsic.clone())

        intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space
        proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]

        proj_mats += [proj_mat_l @ ref_proj_inv]


        imgs = torch.stack(imgs)
        c2ws = torch.stack(c2ws)
        w2cs = torch.stack(w2cs)
        proj_mats = torch.stack(proj_mats)
        intrinsics = torch.stack(intrinsics)
        depths_h = torch.zeros(self.n_views+1, 1, 1)
        near_fars = torch.tensor(depth_range)[None].repeat(self.n_views+1, 1)

        #import pdb; pdb.set_trace()
        return {
            'images': imgs,
            'depths_h': depths_h,
            'w2cs': w2cs,
            'c2ws': c2ws,
            'near_fars': near_fars,
            'proj_mats': proj_mats,
            'intrinsics': intrinsics,
        }


    def read_source_views(self, pair_idx=None, device=torch.device("cpu")):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # load full resolution image then resize
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        print('original focal', focal)

        focal = [focal* self.img_wh[0] / W, focal* self.img_wh[1] / H]
        print('porcessed focal', focal)

        # Step 2: correct poses
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses, _ = center_poses(poses, self.blender2opencv)
        # poses = poses @ self.blender2opencv

        # sub select training views from pairing file
        if pair_idx is None:
            name = os.path.basename(self.root_dir)
            pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]

            # positions = poses[pair_idx,:3,3]
            # dis = np.sum(np.abs(positions - np.mean(positions, axis=0, keepdims=True)), axis=-1)
            # pair_idx = [pair_idx[i] for i in np.argsort(dis)[:3]]
            print(f'====> ref idx: {pair_idx}')


        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        src_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

        w, h = self.img_wh

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i, idx in enumerate(pair_idx):
            c2w = torch.eye(4).float()
            image_path = image_paths[idx]
            c2w[:3] = torch.FloatTensor(poses[idx])
            w2c = torch.inverse(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = torch.eye(4)
            intrinsic = torch.tensor([[focal[0], 0, w / 2], [0, focal[1], h / 2], [0, 0, 1]]).float()
            intrinsics.append(intrinsic.clone())
            intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_l)
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]


            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.stack(c2ws).float().to(device)
        pose_source['w2cs'] = torch.stack(w2cs).float().to(device)
        pose_source['intrinsics'] = torch.stack(intrinsics).float().to(device)


        near_far_source = [bounds[pair_idx].min()*0.8,bounds[pair_idx].max()*1.2]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])
