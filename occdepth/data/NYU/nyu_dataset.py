import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from occdepth.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
import pickle
import torch.nn.functional as F

import math
import skimage
import skimage.io
import skimage.transform


def img_transform(crop, flip):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)

    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran

    return ida_mat


def load_depth(depth_path, maxDepth=10):
    depth = skimage.io.imread(depth_path)
    depth = depth / 8000.0
    depth = depth.astype("float32")
    depth[depth > maxDepth] = maxDepth
    depth[depth < 0] = 0
    return depth


class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
        pattern_id=0,
        use_depth_gt=False,
        use_strong_img_aug=False,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.root = os.path.join(root, "NYU" + split)
        self.preprocess_root = preprocess_root
        self.base_dir = os.path.join(preprocess_root, "base", "NYU" + split)
        self.fliplr = fliplr

        self.with_depth_gt = use_depth_gt

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        # strong_img_aug
        self.do_strong_img_aug = use_strong_img_aug
        print("INFO: Use strong_img_aug: {}".format(self.do_strong_img_aug))
        if self.do_strong_img_aug:
            self.strong_img_aug = transforms.Compose(
                [
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomErasing(
                        scale=(0.01, 0.02), ratio=(0.3, 3), value=0
                    ),
                ]
            )
            self.blur_img_aug = transforms.Compose(
                [
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ]
            )

        self.scan_names = glob.glob(os.path.join(self.root, "*.bin"))

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.pattern_id = pattern_id

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, name + ".pkl")

        with open(filepath, "rb") as handle:
            data = pickle.load(handle)

        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)

        # lss config
        data["T_velo_2_cam"] = T_world_2_cam[
            np.newaxis, ...
        ]  # TODO, velo change to world
        data["num_views"] = 1

        vox_origin = data["voxel_origin"]

        # data["cam_k"] = self.cam_k
        data["cam_k"] = np.array([self.cam_k])
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        data["target"] = target
        target_1_4 = data["target_1_16"]

        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.voxel_size,
            self.img_W,
            self.img_H,
            self.scene_size,
            self.pattern_id,
        )

        projected_pix = projected_pix[np.newaxis, ...]
        fov_mask = fov_mask[np.newaxis, ...]
        pix_z = pix_z[np.newaxis, ...]
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_W,
            self.img_H,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        img = Image.open(rgb_path).convert("RGB")

        # gt_depth
        gt_depth = None
        if self.with_depth_gt:
            stereo_depth_path = os.path.join(os.path.join(self.root, name + ".png"))
            gt_depth = load_depth(stereo_depth_path)
            gt_depth = [gt_depth]  # crop depth

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        # randomly fliplr the image
        flip_flag = False
        crop = (0, 0, 640, 480)
        data["ida_mat"] = []
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            data["projected_pix_1"][..., 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][..., 0]
            )
            flip_flag = True
            # Fliplr the depth
            if self.with_depth_gt:
                gt_depth[0] = np.ascontiguousarray(np.fliplr(gt_depth[0]))
        ida_mat = img_transform(crop, flip_flag)
        data["ida_mat"].append(ida_mat)
        data["ida_mat"] = torch.stack(data["ida_mat"]).numpy()

        # strong img aug
        img = self.normalize_rgb(img)
        if self.do_strong_img_aug:
            if np.random.rand() < 0.3:
                img = self.blur_img_aug(img)
            if np.random.rand() < 0.3:
                img = self.strong_img_aug(img)
        data["img"] = img.unsqueeze(0)

        if self.with_depth_gt:
            gt_depth = torch.as_tensor(np.stack(gt_depth))
            data["gt_depth"] = gt_depth

        return data

    def __len__(self):
        return len(self.scan_names)
