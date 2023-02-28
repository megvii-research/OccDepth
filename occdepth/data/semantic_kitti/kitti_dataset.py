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
import math
import skimage
import skimage.io
import skimage.transform
from occdepth.data.semantic_kitti.io_data import _read_occluded_SemKITTI


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


def load_depth(depth_path, scale=256):
    depth = skimage.io.imread(depth_path)
    depth = depth.astype("float32")
    depth[depth > 0] = depth[depth > 0] / scale
    return depth


class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
        pattern_id=1,
        multi_view_mode=False,
        use_stereo_depth_gt=False,
        use_lidar_depth_gt=False,
        data_stereo_depth_root=None,
        data_lidar_depth_root=None,
        with_occluded=False,
        use_strong_img_aug=False,
    ):
        super().__init__()
        self.root = root
        self.label_root = os.path.join(preprocess_root, "labels")
        self.n_classes = 20
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split
        self.sequences = splits[split]
        # local patch ->local frustum, for frustum loss
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = math.ceil(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.fliplr = fliplr

        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1220  # 1216
        self.img_H = 370  # 352
        self.pattern_id = pattern_id
        self.multi_view_mode = multi_view_mode

        # depth gt
        self.use_stereo_depth_gt = use_stereo_depth_gt
        self.use_lidar_depth_gt = use_lidar_depth_gt
        self.with_depth_gt = self.use_stereo_depth_gt or self.use_lidar_depth_gt
        self.data_stereo_depth_root = data_stereo_depth_root
        self.data_lidar_depth_root = data_lidar_depth_root

        # occluded
        self.with_occluded = with_occluded

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

        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = [calib["P2"]]
            P.append(calib["P3"])
            P = np.array(P)
            Tr_velo_2_cam_0 = calib["Tr"]
            proj_matrix = [P[0] @ Tr_velo_2_cam_0]
            proj_matrix.append(P[1] @ Tr_velo_2_cam_0)
            proj_matrix = np.array(proj_matrix)

            cam_k2 = P[0][0:3, 0:3]
            cam_k3 = P[1][0:3, 0:3]

            # Fix external parameter transformation bug
            T_velo_2_cam2 = np.identity(4)  # 4x4 matrix
            T_velo_2_cam2[:3, :4] = np.linalg.inv(cam_k2) @ proj_matrix[0]

            # Transform from lidar to cam3
            T_velo_2_cam3 = np.identity(4)  # 4x4 matrix
            T_velo_2_cam3[:3, :4] = np.linalg.inv(cam_k3) @ proj_matrix[1]

            T_velo_2_cam = [T_velo_2_cam2, T_velo_2_cam3]
            T_velo_2_cam = np.array(T_velo_2_cam)

            glob_path = os.path.join(
                self.root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )
            for voxel_path in glob.glob(glob_path):
                self.scans.append(
                    {
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        """
        Return:
            data(Dict): {
                "frame_id"(int): frame_id,
                "sequence"(str): sequence,
                "P"(array, 3x4): P,
                "T_velo_2_cam"(array, 4x4): T_velo_2_cam,
                "proj_matrix"(array, 3x4): proj_matrix,
                "projected_pix_{scale_3d}"(array, Nx2):
                    projected_pix in 2d
                "pix_z_{scale_3d}"(array, N):
                    depth for every 2d points
                "fov_mask_{scale_3d}"(array, N):
                    mask show if exist in cam fov
                "target"(array, HWD):
                    target voxel(256, 256, 32), 0 invalid 1-19 class and 255 invalid
                "CP_mega_matrix(array, HWD, HWD/8)":
                    (4, 4096, 512) supervoxel -> voxel attention 0 or 1
                "frustums_masks"(array, n_frustums^2, W, D, H):
                    compute local frustum voxel mask for patch
                "frustums_class_dists"(array, n_frustums^2, n_classes):
                    class numbers in each local frustum
            }
        """
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]  # 3x4 intrinsic matrix
        T_velo_2_cam = scan["T_velo_2_cam"]  # 4x4 lidar to cam matrix
        proj_matrix = scan["proj_matrix"]  # 3x4 lidar to pixel

        if self.multi_view_mode:
            pass
        else:
            T_velo_2_cam = T_velo_2_cam[0][np.newaxis, ...]

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = [
            os.path.join(
                self.root,
                "dataset",
                "sequences",
                sequence,
                "image_2",
                frame_id + ".png",
            )
        ]

        rgb_path.append(
            os.path.join(
                self.root,
                "dataset",
                "sequences",
                sequence,
                "image_3",
                frame_id + ".png",
            )
        )

        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
            "num_views": 2 if self.multi_view_mode else 1,
        }
        # TODO check only self.project_scale is enough?
        scale_3ds = [self.output_scale, self.project_scale]

        data["scale_3ds"] = scale_3ds
        if self.multi_view_mode:
            cam_k = np.array([P[0][0:3, 0:3], P[1][0:3, 0:3]])
        else:
            cam_k = np.array([P[0][0:3, 0:3]])
        data["cam_k"] = cam_k

        for scale_3d in scale_3ds:
            data["projected_pix_{}".format(scale_3d)] = []
            data["pix_z_{}".format(scale_3d)] = []
            data["fov_mask_{}".format(scale_3d)] = []

        for idx_view in range(data["num_views"]):
            for scale_3d in scale_3ds:
                # compute the 3D-2D mapping
                projected_pix, fov_mask, pix_z = vox2pix(
                    T_velo_2_cam[idx_view],
                    cam_k[idx_view],
                    self.vox_origin,
                    self.voxel_size * scale_3d,
                    self.img_W,
                    self.img_H,
                    self.scene_size,
                    self.pattern_id,
                )
                data["projected_pix_{}".format(scale_3d)].append(projected_pix)
                data["pix_z_{}".format(scale_3d)].append(pix_z)
                data["fov_mask_{}".format(scale_3d)].append(fov_mask)

        for scale_3d in scale_3ds:
            data["projected_pix_{}".format(scale_3d)] = np.array(
                data["projected_pix_{}".format(scale_3d)]
            )
            data["pix_z_{}".format(scale_3d)] = np.array(
                data["pix_z_{}".format(scale_3d)]
            )
            data["fov_mask_{}".format(scale_3d)] = np.array(
                data["fov_mask_{}".format(scale_3d)]
            )

        if self.split != "test":
            target_1_path = os.path.join(
                self.label_root, sequence, frame_id + "_1_1.npy"
            )
            target = np.load(target_1_path)
            # 0 means unlabeled, 1-19 means class, 255 means invalid in target
            # see in `occdepth/data/semantic_kitti/semantic-kitti` learning_map
            data["target"] = target
            target_8_path = os.path.join(
                self.label_root, sequence, frame_id + "_1_8.npy"
            )
            target_1_8 = np.load(target_8_path)

            # compute supervoxel -> voxel attention matrix(4, HWD, H/2 W/2 D/2)
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            data["CP_mega_matrix"] = CP_mega_matrix

        if self.with_occluded:
            occluded_path = os.path.join(
                self.root,
                "dataset",
                "sequences",
                sequence,
                "voxels",
                frame_id + ".occluded",
            )
            occluded = _read_occluded_SemKITTI(occluded_path)
            occluded = occluded.reshape(256, 256, 32)

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data["pix_z_{}".format(self.output_scale)]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=20,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        # gt_depth
        gt_depth = None
        if self.split != "test":
            if self.use_stereo_depth_gt:
                stereo_depth_path = os.path.join(
                    self.data_stereo_depth_root,
                    "dataset",
                    "sequences",
                    sequence,
                    "depth",
                    frame_id + ".png",
                )
                gt_depth = load_depth(stereo_depth_path)
                gt_depth = [gt_depth[: self.img_H, : self.img_W]]  # crop depth
            elif self.use_lidar_depth_gt:
                lidar_depth_path = [
                    os.path.join(
                        self.data_lidar_depth_root,
                        "dataset",
                        "sequences",
                        sequence,
                        "lidar_depth",
                        frame_id,
                        str(i) + ".npy",
                    )
                    for i in range(data["num_views"])
                ]
                gt_depth = [
                    np.load(lidar_depth_path[i])[: self.img_H, : self.img_W]
                    for i in range(len(lidar_depth_path))
                ]

        data["img"] = []
        data["ida_mat"] = []
        fliplr_random = np.random.rand()
        for i in range(data["num_views"]):
            img_i = Image.open(rgb_path[i]).convert("RGB")

            # Image augmentation
            if self.color_jitter is not None:
                img_i = self.color_jitter(img_i)

            # PIL to numpy
            img_i = np.array(img_i, dtype=np.float32, copy=False) / 255.0
            img_i = img_i[: self.img_H, : self.img_W, :]  # crop image
            crop = (0, 0, self.img_W, self.img_H)

            # Fliplr the image
            flip_flag = False
            if fliplr_random < self.fliplr:
                img_i = np.ascontiguousarray(np.fliplr(img_i))
                for scale in scale_3ds:
                    key = "projected_pix_" + str(scale)
                    data[key][i][:, :, 0] = img_i.shape[1] - 1 - data[key][i][:, :, 0]
                flip_flag = True
            ida_mat = img_transform(crop, flip_flag)

            # Fliplr the depth
            if self.split != "test":
                if self.with_depth_gt and fliplr_random < self.fliplr:
                    if self.use_stereo_depth_gt and i > 0:
                        # only left img has stereo depth
                        pass
                    else:
                        gt_depth[i] = np.ascontiguousarray(np.fliplr(gt_depth[i]))

            # strong img aug
            img_i = self.normalize_rgb(img_i)
            if self.do_strong_img_aug:
                if np.random.rand() < 0.3:
                    img_i = self.blur_img_aug(img_i)
                if np.random.rand() < 0.3:
                    img_i = self.strong_img_aug(img_i)

            data["img"].append(img_i)
            data["ida_mat"].append(ida_mat)
        data["img"] = torch.stack(data["img"])
        data["ida_mat"] = torch.stack(data["ida_mat"]).numpy()

        if self.split != "test":
            if self.with_depth_gt:
                gt_depth = torch.as_tensor(np.stack(gt_depth))
                data["gt_depth"] = gt_depth

        if self.with_occluded:
            occluded = torch.as_tensor(np.stack(occluded))
            data["occluded"] = occluded

        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out


if __name__ == "__main__":
    import hydra

    @hydra.main(config_name="../../config/occdepth.yaml")
    def test(config):
        ds = KittiDataset(
            split="train",
            root="/data/dataset/KITTI_Odometry_Semantic",
            preprocess_root="/data/dataset/kitti_semantic_preprocess",
            project_scale=config.project_scale,
            frustum_size=config.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )
        """
        Args:
            frame_id(str)
            sequence(str)
            P(array, 4x4)
            T_velo_2_cam(array, 4x4)
            proj_matrix(array, 4x4)
            scale_3ds(list)
            cam_k(array, 3x4)
            projected_pix_1(array, nx2)
            pix_z_1(array, n)
            fov_mask_1(array, n)
            projected_pix_2(array, n2x2)
            pix_z_2(array, n2x2)
            fov_mask_2(array, n2x2)
            target(array, x*y*z): 256x256x32 voxel
            CP_mega_matrix(array, 4x4096x512)
            frustums_masks(64x256x256x32)
            frustums_class_dists(64x20)
            img(array, cxhxw): 3x370x1220
        """
        d = ds[0]

    test()
    from IPython import embed

    embed()
