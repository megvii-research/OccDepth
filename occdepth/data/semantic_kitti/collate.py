import torch


def collate_fn(batch, with_depth_gt=False, with_occluded=False):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    T_velo_2_cams = []
    ida_mats = []
    gt_depths = []
    occludeds = []
    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        ida_mats.append(torch.from_numpy(input_dict["ida_mat"]).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict and input_dict["frustums_masks"] is not None:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)
        if (
            with_depth_gt
            and "gt_depth" in input_dict
            and input_dict["gt_depth"] is not None
        ):
            gt_depth = input_dict["gt_depth"]
            gt_depths.append(gt_depth)

        if with_occluded and "occluded" in input_dict:
            occluded = input_dict["occluded"]
            occludeds.append(occluded)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])

        if "target" in input_dict:
            target = torch.from_numpy(input_dict["target"])
            targets.append(target)
        if "CP_mega_matrix" in input_dict:
            CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "ida_mats": ida_mats,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
    }
    if "target" in input_dict:
        ret_data["target"] = torch.stack(targets)

    if with_depth_gt and len(gt_depths) > 0:
        ret_data["gt_depth"] = torch.stack(gt_depths)

    if with_occluded and len(occludeds) > 0:
        ret_data["occluded"] = torch.stack(occludeds)

    for key in data:
        ret_data[key] = data[key]
    return ret_data
