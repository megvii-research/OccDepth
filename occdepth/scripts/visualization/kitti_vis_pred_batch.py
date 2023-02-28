import os
import pickle
import numpy as np

from omegaconf import DictConfig
import hydra


import cv2
from tqdm import tqdm


from numpy import array

from mayavi import mlab
from mayavi.api import Engine

engine = Engine()

engine.start()


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    if len(fov_mask.shape) == 3:
        fov_mask = fov_mask[0, :, 0]

    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    print("len(engine.scenes): ", len(engine.scenes))

    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    scene = engine.scenes[0]
    scene.scene.camera.position = [
        -60.58825795977401,
        21.157014981545093,
        50.718208823185556,
    ]
    scene.scene.camera.focal_point = [
        26.360000076238066,
        25.800000071525574,
        1.9100000262260437,
    ]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [
        0.490062445498992,
        -0.015325179458712417,
        0.8715526021911161,
    ]
    scene.scene.camera.clipping_range = [49.30213893945163, 161.93546276257854]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    if False:
        mlab.show()
    else:
        img = mlab.screenshot()
        mlab.close()
        return img


@hydra.main(config_path=None)
def main(config: DictConfig):

    pkl_dir = "/data/github_code/OccDepth/output/kitti/08"
    img_dir = "/data/dataset/KITTI_Odometry_Semantic/dataset/08/image_2"
    dump_dir = (
        "/data/occupancynet/OccDepth/baseline_v100/dump_dir/baseline_v2"
    )
    if "img_dir" in config:
        img_dir = config["img_dir"]
    if "pkl_dir" in config:
        pkl_dir = config["pkl_dir"]
    if "dump_dir" in config:
        dump_dir = config["dump_dir"]
    os.makedirs(dump_dir, exist_ok=True)

    pkl_names = os.listdir(pkl_dir)
    pkl_names = sorted(pkl_names)
    for idx, pkl_name in tqdm(enumerate(pkl_names)):
        if idx > 2000:
            break
        img_name = pkl_name.replace(".pkl", ".png")
        pkl_path = os.path.join(pkl_dir, pkl_name)
        img_path = os.path.join(img_dir, img_name)

        # scan = config.file
        with open(pkl_path, "rb") as handle:
            b = pickle.load(handle)

        fov_mask_1 = b["fov_mask_1"]
        T_velo_2_cam = b["T_velo_2_cam"]
        vox_origin = np.array([0, -25.6, -2])

        y_pred = b["y_pred"]

        if config.dataset == "kitti_360":
            # Visualize KITTI-360
            draw(
                y_pred,
                T_velo_2_cam,
                vox_origin,
                fov_mask_1,
                voxel_size=0.2,
                f=552.55426,
                img_size=(1408, 376),
                d=7,
            )
        else:
            # Visualize Semantic KITTI
            vox_img = draw(
                y_pred,
                T_velo_2_cam,
                vox_origin,
                fov_mask_1,
                img_size=(1220, 370),
                f=707.0912,
                voxel_size=0.2,
                d=7,
            )
        img = cv2.imread(img_path)
        new_w = img.shape[1]
        new_h = int(new_w / vox_img.shape[1] * vox_img.shape[0])
        vox_img = cv2.resize(vox_img, (new_w, new_h))[:, :, ::-1]

        dump_path = os.path.join(dump_dir, "{:05d}.png".format(idx))
        dump_img = np.concatenate([img, vox_img], axis=0)
        cv2.imwrite(dump_path, dump_img)
        print(f"Dump img: {dump_path}")

    print("done")


if __name__ == "__main__":
    main()
