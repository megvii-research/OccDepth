import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from occdepth.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
from scipy.spatial.transform import Rotation as R


class TartanAirDataset(Dataset):
    def __init__(
        self,
        split,
        config,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        self.root = config.data_dir_root
        self.label_root = os.path.join(config.data_preprocess_root, "labels")
        self.n_classes = config.n_classes
        self.scene = config.scene  # office
        self.scene_diffculty = config.scene_diffculty  # "Easy"
        splits = {
            "train": ["P000", "P001", "P002", "P003", "P004", "P006"],
            "val": ["P005"],
        }
        self.split = split
        self.sequences = splits[split]
        # local patch ->local frustum, for frustum loss
        self.frustum_size = config.frustum_size
        self.scene_size = (
            np.array(config.full_scene_size).reshape(3) * config.voxel_size
        )
        self.fliplr = fliplr
        self.n_relations = config.n_relations

        self.voxel_size = config.voxel_size  # 0.1m
        self.img_W = config.img_W
        self.img_H = config.img_H
        self.pattern_id = config.pattern_id

        self.color_jitter = (
            transforms.ColorJitter(*(color_jitter)) if color_jitter else None
        )
        self.scans = []
        for sequence in self.sequences:
            poses = self.read_poses(
                os.path.join(
                    self.root,
                    self.scene,
                    self.scene_diffculty,
                    sequence,
                    "pose_left.txt",
                ),
                os.path.join(
                    self.root,
                    self.scene,
                    self.scene_diffculty,
                    sequence,
                    "pose_right.txt",
                ),
            )
            intrinsics = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1]])
            T_cam0_2_cams = [
                np.identity(4)
            ]  ## coordinate of depth map0 is the same as img0

            cam_k0 = intrinsics
            cam_k1 = intrinsics
            cam_k = np.array([cam_k0, cam_k1])
            # TODO: need to transform from local frame to cam1
            # world in NED
            T_cam_2_body = np.array(
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            )
            T_cam0_2_world = poses[0, 0] @ T_cam_2_body
            T_cam1_2_world = poses[1, 0] @ T_cam_2_body

            # stereo_baseline = 0.25 :https://github.com/castacks/tartanair_tools/blob/master/data_type.md#depth-image
            T_cam0_2_cams.append(np.linalg.inv(T_cam1_2_world) @ T_cam0_2_world)
            T_cam0_2_cams = np.array(T_cam0_2_cams)

            glob_path = os.path.join(
                self.label_root,
                self.scene,
                self.scene_diffculty,
                sequence,
                "voxels_left",
                "*.pkl",
            )

            for voxel_path in glob.glob(glob_path):
                self.scans.append(
                    {
                        "sequence": sequence,
                        "cam_k": cam_k,
                        "T_cam0_2_cams": T_cam0_2_cams,
                        "voxel_path": voxel_path,
                        "poses": poses,
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
        cam_k = scan["cam_k"]  # 3x3 intrinsic matrix
        T_cam0_2_cams = scan["T_cam0_2_cams"]  # 4x4 lidar to cam matrix

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = [
            os.path.join(
                self.root,
                self.scene,
                self.scene_diffculty,
                sequence,
                "image_left",
                frame_id + "_left.png",
            )
        ]

        rgb_path.append(
            os.path.join(
                self.root,
                self.scene,
                self.scene_diffculty,
                sequence,
                "image_right",
                frame_id + "_right.png",
            )
        )
        with open(voxel_path, "rb") as handle:
            pickl_data = pickle.load(handle)

        target = pickl_data["target_1_1"]
        target_1_4 = pickl_data["target_1_4"]
        self.vox_origin = pickl_data["vox_origin"]
        T_voxel_2_cam = pickl_data["T_velo_2_cam"]
        T_velo_2_cam = []
        for i in range(len(T_cam0_2_cams)):
            T_velo_2_cam.append(T_cam0_2_cams[i] @ T_voxel_2_cam)
        T_velo_2_cam = np.array(T_velo_2_cam)
        # 0 means unlabeled, 1-19 means class, 255 means invalid in target
        # see in `occdepth/data/semantic_kitti/semantic-kitti` learning_map
        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "cam_k": cam_k,
            "T_velo_2_cam": T_velo_2_cam,
            "num_views": 2,
            "target": target,
            "target_1_4": target_1_4,
            "vox_origin": self.vox_origin,
        }
        # compute supervoxel -> voxel attention matrix(4, HWD, H/2 W/2 D/2)
        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix

        data["projected_pix_1"] = []
        data["pix_z_1"] = []
        data["fov_mask_1"] = []

        for idx_view in range(data["num_views"]):
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam[idx_view],
                cam_k[idx_view],
                self.vox_origin,
                self.voxel_size,
                self.img_W,
                self.img_H,
                self.scene_size - 0.0001,  # for np.ceil in vox2pix
                self.pattern_id,
            )
            data["projected_pix_1"].append(projected_pix)
            data["pix_z_1"].append(pix_z)
            data["fov_mask_1"].append(fov_mask)

        data["projected_pix_1"] = np.array(data["projected_pix_1"])
        data["pix_z_1"] = np.array(data["pix_z_1"])
        data["fov_mask_1"] = np.array(data["fov_mask_1"])

        # Compute the masks, each indicate the voxels of a local frustum
        projected_pix_output = data["projected_pix_1"]
        pix_z_output = data["pix_z_1"]
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix_output,
            pix_z_output,
            target,
            self.img_W,
            self.img_H,
            dataset="tartanair",
            n_classes=14,
            size=self.frustum_size,
        )

        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        data["img"] = []
        fliplr_flag = np.random.rand()
        for i in range(data["num_views"]):
            img_i = Image.open(rgb_path[i]).convert("RGB")

            # Image augmentation
            if self.color_jitter is not None:
                img_i = self.color_jitter(img_i)

            # PIL to numpy
            img_i = np.array(img_i, dtype=np.float32, copy=False) / 255.0
            img_i = img_i[: self.img_H, : self.img_W, :]  # crop image

            # Fliplr the image
            if fliplr_flag < self.fliplr:
                img_i = np.ascontiguousarray(np.fliplr(img_i))
                key = "projected_pix_1"
                data[key][i][:, :, 0] = img_i.shape[1] - 1 - data[key][i][:, :, 0]

            data["img"].append(self.normalize_rgb(img_i))
        data["img"] = torch.stack(data["img"])
        return data

    def __len__(self):
        return len(self.scans)

    def read_poses(self, pose_path0, pose_path1):
        """Load ground truth poses (T_w_cam0,T_w_cam1) from file."""
        # Read and parse the poses
        poses0 = []
        poses1 = []
        with open(pose_path0, "r") as f:
            lines0 = f.readlines()
            for line in lines0:  # xyz+q,GT frame (NED)
                pos_quat0 = np.fromstring(line, dtype=float, sep=" ")
                T_w_cam0 = self.pos_quat2SE(pos_quat0)
                poses0.append(T_w_cam0)

        with open(pose_path1, "r") as f:
            lines1 = f.readlines()
            for line in lines1:
                pos_quat1 = np.fromstring(line, dtype=float, sep=" ")
                T_w_cam1 = self.pos_quat2SE(pos_quat1)
                poses1.append(T_w_cam1)

        return np.array([poses0, poses1])

    def pos_quat2SE(self, quat_data):
        """
        Modify from  https://github.com/castacks/tartanair_tools/blob/master/evaluation/transformation.py
        """
        SO = R.from_quat(quat_data[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3] = quat_data[0:3]
        return SE


if __name__ == "__main__":
    import hydra

    @hydra.main(config_name="../../config/occdepth.yaml")
    def test(config):
        ds = TartanAirDataset(
            split="train",
            config=config,
        )
        """
        Args:
            frame_id(str)
            sequence(str)
            P(array, 4x4)
            T_velo_2_cam(array, 4x4)
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
