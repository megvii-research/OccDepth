final_dim = (370, 1220)
flosp_depth_conf = {
    "x_bound": [0, 51.2, 0.2],
    "y_bound": [-25.6, 25.6, 0.2],
    "z_bound": [-2, 4.4, 0.2],
    "d_bound": [2.0, 54.0, 0.5],
    "final_dim": final_dim,
    "output_channels": 64,
    "downsample_factor": 8,
    "depth_net_conf": dict(in_channels=64, mid_channels=128),
    "disc_cfg": dict(mode="LID"),  # LID or UD
    "agg_voxel_mode": "mean",  # mean or sum
}
