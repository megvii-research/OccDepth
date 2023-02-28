import torch


def collate_fn(batch):
    data = {}
    imgs = []
    CP_mega_matrices = []
    targets = []
    frame_ids = []
    sequences = []

    cam_ks = []
    vox_origins = []
    T_velo_2_cams = []
    frustums_masks = []
    frustums_class_dists = []

    data["projected_pix_1"] = []
    data["fov_mask_1"] = []

    for idx, input_dict in enumerate(batch):
        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "vox_origin" in input_dict:
            vox_origins.append(torch.from_numpy(input_dict["vox_origin"]))
        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        img = input_dict["img"]
        imgs.append(img)

        frame_ids.append(input_dict["frame_id"])
        sequences.append(input_dict["sequence"])

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))

    ret_data = {
        "frame_id": frame_ids,
        "sequence": sequences,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "vox_origin": vox_origins,
        "img": torch.stack(imgs),
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets),
    }

    for key in data:
        ret_data[key] = data[key]
    return ret_data
