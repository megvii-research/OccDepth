import torch


def collate_fn(batch, with_depth_gt=False):
    data = {}
    imgs = []
    targets = []
    names = []
    cam_poses = []

    gt_depths = []

    vox_origins = []
    cam_ks = []
    T_velo_2_cams = []
    ida_mats = []

    CP_mega_matrices = []

    data["projected_pix_1"] = []
    data["fov_mask_1"] = []
    data["frustums_masks"] = []
    data["frustums_class_dists"] = []

    for idx, input_dict in enumerate(batch):
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))
        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        ida_mats.append(torch.from_numpy(input_dict["ida_mat"]).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(input_dict["voxel_origin"]).double())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        if with_depth_gt and "gt_depth" in input_dict:
            gt_depth = input_dict["gt_depth"]
            gt_depths.append(gt_depth)

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)

    ret_data = {
        "CP_mega_matrices": CP_mega_matrices,
        "cam_pose": torch.stack(cam_poses),
        # "cam_k": torch.stack(cam_ks),
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "ida_mats": ida_mats,
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
        "target": torch.stack(targets),
    }

    if with_depth_gt and len(gt_depths) > 0:
        ret_data["gt_depth"] = torch.stack(gt_depths)

    for key in data:
        ret_data[key] = data[key]
    return ret_data
