import numpy as np
import os, shutil
import glob
import pickle
from tqdm import tqdm
import cv2
from occdepth.data.tartanair.params import class_colors
from occdepth.data.tartanair.params import tartanair_class_dict
from scipy.spatial.transform import Rotation as R
from numba import njit, jit
import multiprocessing
from numba.typed import Dict

scenes = ["office"]
difficultys = ["Easy", "Hard"]
sequences = {
    "Easy": ["P000", "P001", "P002", "P003", "P004", "P005", "P006"],
    "Hard": ["P000", "P001", "P002", "P003", "P004", "P005", "P006", "P007"],
}
data_root_dir = "/data/dataset/TartanAir_full"
preprocess_dir = "/data/dataset/TartanAir_full_preprocess"
labels_dir = "/data/dataset/TartanAir_full_preprocess/labels"

scene = None
scene_diffculty = None
sequence = None
poses = None

intrinsics = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1]])
# pose = CameraPoseTransform.get_matrix_from_pose(frame_data["pose_left"])
pose = np.eye(4)
# pose[:3, :3] = np.array(
#     [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
# )  # consistent with semantic_kitti
vox_origin = np.array([-6, -3, 0])  # cam coord
vox_shape = (120, 48, 120)

tartanair_class_dict_nj = Dict()
for k, v in tartanair_class_dict.items():
    tartanair_class_dict_nj[k] = v


@jit(nopython=True)
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(x.size):
        for j in range(y.size):
            xx[i, j] = j
            yy[i, j] = i
    return xx, yy


@jit(nopython=True)
def find_new_seg(seg_id, class_dicts):
    new_id = -1
    for key in class_dicts:
        pair_data = class_dicts[key]
        if seg_id in pair_data[1]:
            new_id = pair_data[0]
    if new_id < 0:
        new_id = len(class_dicts) - 1
    return new_id


def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


@jit(nopython=True)
def depth2voxel(depth, seg, cam_pose, vox_origin, cam_k, voxel_size, unit, class_dicts):
    # voxel_size: (240, 240, 80), unit: 0.04
    # ---- Get point in camera coordinate
    H, W = depth.shape
    class_num = len(class_dicts)

    gx, gy = meshgrid(np.arange(H), np.arange(W))
    pt_cam = np.zeros((H, W, 3), dtype=np.float64)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter

    # ---- Get point in lidar coordinate
    # pt_lidar = (cam_pose[:3, :3]@pt_cam.reshape(-1, 3).T).T.reshape(pt_cam.shape) + cam_pose[:3, 3]
    pt_lidar = cam_pose[:3, :3] @ pt_cam.reshape(-1, 3).T
    pt_lidar = pt_lidar.reshape(3, pt_cam.shape[0], pt_cam.shape[1])
    pt_lidar = pt_lidar.transpose(1, 2, 0) + cam_pose[:3, 3]
    pt_vox = pt_lidar - vox_origin

    # ---- Lidar coordinate to grid/voxel coordinate
    point_grid = pt_vox / unit  # Get point in grid coordinate, each grid is a voxel
    point_grid = np.rint(point_grid).astype(np.int32)
    voxel_binary = np.zeros(
        (voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.uint8
    )  # (W, H, D)
    voxel_cls = np.zeros(
        (voxel_size[0], voxel_size[1], voxel_size[2]), dtype=np.uint8
    )  # (W, H, D)

    voxel_cnt = np.zeros(
        (voxel_size[0], voxel_size[1], voxel_size[2], class_num), dtype=np.int32
    )  # (W, H, D, clas_num)

    for h in range(H):
        for w in range(W):
            i_x, i_y, i_z = point_grid[h, w, :]
            if (
                0 <= i_x < voxel_size[0]
                and 0 <= i_y < voxel_size[1]
                and 0 <= i_z < voxel_size[2]
            ):
                voxel_binary[i_x, i_y, i_z] = 1
                new_seg_id = find_new_seg(seg[h, w], class_dicts)  # remap class_id
                voxel_cnt[i_x, i_y, i_z, new_seg_id] += 1
    for h in range(H):
        for w in range(W):
            i_x, i_y, i_z = point_grid[h, w, :]
            if (
                0 <= i_x < voxel_size[0]
                and 0 <= i_y < voxel_size[1]
                and 0 <= i_z < voxel_size[2]
            ):
                # voxel_cls[i_x, i_y, i_z] = seg[h, w] + 1  # 0表示无class
                voxel_cls[i_x, i_y, i_z] = np.argmax(
                    voxel_cnt[i_x, i_y, i_z]
                )  # save the  class that has max count number

    return voxel_binary, voxel_cls  # (W, H, D), (W, H, D)


def export_voxels_sub(depth_path):
    filename = os.path.basename(depth_path)
    frame_id = filename.split("_")[0]
    if int(frame_id) % 5 == 0:
        pose_c2w = poses[int(frame_id)]
        depth_img = np.load(depth_path)
        seg_path = os.path.join(
            data_root_dir,
            scene,
            scene_diffculty,
            sequence,
            "seg_left",
            frame_id + "_left_seg.npy",
        )
        seg_img = np.load(seg_path)

        voxel_binary, voxel_cls = depth2voxel(
            depth=depth_img,
            seg=seg_img,
            cam_pose=pose_c2w,
            vox_origin=vox_origin,
            cam_k=intrinsics,
            voxel_size=vox_shape,
            unit=0.1,  # 0.1m
            class_dicts=tartanair_class_dict_nj,
        )
        seq_label_dir = os.path.join(labels_dir, scene, scene_diffculty, sequence)
        target_path = os.path.join(seq_label_dir, "voxels_left", frame_id + ".pkl")

        out_dict = {}
        out_dict["vox_origin"] = vox_origin
        out_dict["cam_k"] = intrinsics
        out_dict["T_velo_2_cam"] = np.linalg.inv(pose_c2w)
        out_dict["fov_mask_1_1"] = voxel_cls.reshape(-1) > 0
        out_dict["target_1_1"] = voxel_cls
        voxel_cls_1_4 = _downsample_label(voxel_cls, vox_shape, 4)
        out_dict["target_1_4"] = voxel_cls_1_4
        out_dict["fov_mask_1_4"] = voxel_cls_1_4.reshape(-1) > 0

        with open(target_path, "wb") as f:
            pickle.dump(out_dict, f)


def export_voxels():

    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    global scene
    global scene_diffculty
    global sequence
    global poses
    for scene_i in scenes:
        for scene_diffculty_i in tqdm(difficultys):
            for sequence_i in tqdm(sequences[scene_diffculty_i]):
                scene = scene_i
                scene_diffculty = scene_diffculty_i
                sequence = sequence_i

                seq_label_dir = os.path.join(
                    labels_dir, scene, scene_diffculty, sequence
                )
                os.makedirs(seq_label_dir, exist_ok=True)
                os.makedirs(os.path.join(seq_label_dir, "voxels_left"), exist_ok=True)
                poses = read_pitch_poses(
                    os.path.join(
                        data_root_dir, scene, scene_diffculty, sequence, "pose_left.txt"
                    ),
                )
                depth_paths = os.path.join(
                    data_root_dir,
                    scene,
                    scene_diffculty,
                    sequence,
                    "depth_left",
                    "*.npy",
                )
                depth_paths = glob.glob(depth_paths)
                depth_paths.sort()
                pool = multiprocessing.Pool(processes=8)
                # pool.map(export_voxels_sub, depth_paths)
                r = list(
                    tqdm(
                        pool.imap(export_voxels_sub, depth_paths),
                        total=len(depth_paths),
                        desc="export_voxels",
                    )
                )
                pool.close()
                pool.join()


def pos_quat2SE(quat_data):
    """
    Modify from  https://github.com/castacks/tartanair_tools/blob/master/evaluation/transformation.py
    """
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.eye(4)
    SE[0:3, 0:3] = SO
    SE[0:3, 3] = quat_data[0:3]
    return SE


def read_pitch_poses(pose_path0):
    """Load ground truth poses (T_w_body0) from file."""
    # Read and parse the poses
    poses0 = []
    T_body_cam0 = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    with open(pose_path0, "r") as f:
        lines0 = f.readlines()
        for line in lines0:  # xyz+q,GT frame (NED)
            pos_quat0 = np.fromstring(line, dtype=float, sep=" ")
            T_center_body = rollpitch_2SE(
                pos_quat0
            )  # use body center as origin position and remove yaw angle, called center coordinates
            T_center_cam0 = np.linalg.inv(T_body_cam0) @ T_center_body @ T_body_cam0
            poses0.append(T_center_cam0)
    return np.array(poses0)


def rollpitch_2SE(quat_data):
    """
    Modify from  https://github.com/castacks/tartanair_tools/blob/master/evaluation/transformation.py
    """
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    euler_inv = R.from_matrix(np.linalg.inv(SO)).as_euler("zyx", degrees=False)
    euler_inv[0] = 0  # not use yaw
    SO_inv = R.from_euler("zyx", euler_inv, degrees=False).as_matrix()
    SE = np.eye(4)
    SE[0:3, 0:3] = np.linalg.inv(SO_inv)
    return SE


def stat_cls_freqs():

    for scene in scenes:
        for scene_diffculty in tqdm(difficultys):
            seg_cnt = np.zeros(256, dtype=np.int64)  # (W, H, D, 3)

            for sequence in tqdm(sequences[scene_diffculty]):
                print(scene, scene_diffculty, sequence)
                seg_paths = os.path.join(
                    data_root_dir, scene, scene_diffculty, sequence, "seg_left", "*.npy"
                )
                for seg_path in tqdm(glob.glob(seg_paths)):
                    seg_img = np.load(seg_path)
                    cnt = np.bincount(seg_img.flatten())
                    cnt = np.pad(cnt, (0, 256 - len(cnt)), "constant")
                    seg_cnt += cnt
            stat_path = os.path.join(
                data_root_dir, scene, scene_diffculty, "seg_stat.npy"
            )
            print(scene_diffculty, seg_cnt)
            np.save(stat_path, seg_cnt)


# for class labels with human and set tags, so it is used as a testing function
def search_save_view_seg(seq):
    scene = scenes[0]
    scene_diffculty = difficultys[0]
    sequence = sequences[scene_diffculty][seq]
    seg_paths = os.path.join(
        data_root_dir, scene, scene_diffculty, sequence, "seg_left", "*.npy"
    )
    cls_img_path = os.path.join(data_root_dir, scene, scene_diffculty, "cls_img")
    os.makedirs(cls_img_path, exist_ok=True)

    seg_paths = glob.glob(seg_paths)
    seg_paths.sort()
    n_imgs = len(seg_paths)
    record_id_set = set()
    for ith_frame in tqdm(range(n_imgs)):
        img_path = os.path.join(
            data_root_dir,
            scene,
            scene_diffculty,
            sequence,
            "image_left",
            "{:06d}_left.png".format(ith_frame),
        )
        seg_path = seg_paths[ith_frame]
        seg_img = np.load(seg_path)
        H, W = seg_img.shape
        cls_all_img = np.zeros((H, W, 3), dtype=np.uint8)
        all_flag = False
        for idx in range(len(class_colors)):
            cls_img = np.zeros((H, W, 3), dtype=np.uint8)
            select_flag = seg_img == idx
            cls_img[select_flag] = class_colors[idx]
            cls_all_img[select_flag] = class_colors[idx]
            if (
                idx not in record_id_set
                and select_flag.any()
                and len(cls_img[select_flag]) > 1000
            ):
                record_id_set.add(idx)
                all_flag = True
                cv2.imwrite(
                    os.path.join(
                        cls_img_path, "{:03d}_{:06d}_cls.png".format(idx, ith_frame)
                    ),
                    cls_img,
                )
                copy_img_path = os.path.join(
                    cls_img_path, "{:03d}_{:06d}_left.png".format(idx, ith_frame)
                )
                shutil.copyfile(img_path, copy_img_path)
        if all_flag:
            cv2.imwrite(
                os.path.join(cls_img_path, "cls_all_{:06d}.png".format(ith_frame)),
                cls_all_img,
            )


def test():
    global scenes
    global difficultys
    global sequences
    scenes = ["office"]
    difficultys = ["Hard"]
    sequences = {
        "Easy": ["P000"],
        "Hard": ["P000"],
    }
    export_voxels()


if __name__ == "__main__":
    # test()
    export_voxels()
    # stat_cls_freqs()
    # search_save_view_seg(3)
