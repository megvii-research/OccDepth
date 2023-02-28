final_dim = (480, 640)
flosp_depth_conf = {
    "x_bound": [0, 4.8, 0.08],
    "y_bound": [-2.4, 2.4, 0.08],
    "z_bound": [0, 2.88, 0.08],
    "d_bound": [0, 10, 0.08],
    "final_dim": final_dim,
    "output_channels": 64,
    "downsample_factor": 8,
    "depth_net_conf": dict(in_channels=64, mid_channels=128),
    "disc_cfg": dict(mode="LID"),  # LID or UD
    "agg_voxel_mode": "mean",  # mean or sum
}
