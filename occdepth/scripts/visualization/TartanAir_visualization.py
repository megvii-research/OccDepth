import os

os.environ["OPEN3D_CPU_RENDERING"] = "true"
import pickle
import numpy as np
import math
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.rendering as rendering
import cv2  # should import cv2 after open3d
import trimesh
import multiprocessing


width, height = 640, 480
fx, fy, cx, cy = 320, 320, width / 2, height / 2
vertical_fov = 2 * math.atan(0.5 * height / fy) * 180 / np.pi
pkl_dir = ""
img_dir = ""
dump_dir = ""
style = None
pkl_names = None
multi_view_mode = None
gt_view_mode = True


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    # g_xx = np.arange(0, dims[1] + 1)
    # g_yy = np.arange(0, dims[0] + 1)
    # g_zz = np.arange(0, dims[2] + 1 )

    # # Obtaining the grid with coords...
    # xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    xx, yy, zz = np.meshgrid(
        range(dims[0]), range(dims[1]), range(dims[2]), indexing="ij"
    )
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    # temp = np.copy(coords_grid)
    # temp[:, 0] = coords_grid[:, 1]
    # temp[:, 1] = coords_grid[:, 0]
    # coords_grid = np.copy(temp)

    return coords_grid


def get_cube(center, index, size):
    vertices = (
        np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        * size
        * 0.75
        + center
    )

    faces = (
        np.array(
            [
                [4, 7, 3],
                [7, 8, 3],
                [8, 7, 5],
                [7, 6, 5],
                [8, 5, 3],
                [3, 5, 1],
                [3, 1, 4],
                [4, 1, 2],
                [4, 2, 7],
                [7, 2, 6],
                [5, 6, 2],
                [5, 2, 1],
            ]
        )
        + 8 * index
        - 1
    )

    normals = np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0]]
    )
    normals = np.repeat(normals, 2, axis=0)
    return vertices, faces, normals


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    voxel_size=0.1,
    dump_path="",
    style="voxel",
):

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
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

    render = rendering.OffscreenRenderer(width, height, headless=True)
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    render.scene.scene.set_sun_light(
        [0, -1, -1], [1, 1, 1], 6000  # direction  # color
    )  # intensity
    render.scene.scene.enable_sun_light(True)

    colors = (
        np.array(
            [
                [22, 191, 206, 255],
                [214, 38, 40, 255],
                [43, 160, 43, 255],
                [158, 216, 229, 255],
                [114, 158, 206, 255],
                [204, 204, 91, 255],
                [255, 186, 119, 255],
                [147, 102, 188, 255],
                [30, 119, 181, 255],
                [188, 188, 33, 255],
                [255, 127, 12, 255],
                [196, 175, 214, 255],
                [153, 153, 153, 255],
                [150, 240, 80, 255],
                [255, 240, 150, 255],
                [255, 0, 0, 255],
            ]
        )[:, :3]
        / 255.0
    )
    points = fov_voxels[:, :3] + vox_origin
    # points = fov_voxels[:, :3]
    # points[:,1] = -points[:,1]#because up of setup_camera has -1 in y_cam axis,here should flip y_world
    colors = np.stack([colors[int(c)] for c in fov_voxels[:, 3]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if style == "pcd":
        render.scene.add_geometry("pcd", pcd, material)
    elif style == "mesh":
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_nocolor = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=0.2)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh_nocolor.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh_nocolor.faces)),
        )
        mesh = mesh.filter_smooth_laplacian(4)
        mesh.compute_vertex_normals()
        mesh_colors = []
        mesh_vertices = np.asarray(mesh.vertices)
        for vertex in mesh_vertices:
            _, nearest_index, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
            mesh_colors.append(colors[nearest_index[0]])
        mesh_colors = np.stack(mesh_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        render.scene.add_geometry("mesh", mesh, material)
    if style == "pcd2mesh":
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=200)
        )
        pcd.orient_normals_towards_camera_location([0, -10, -60])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([0.25, 0.5])
        )
        render.scene.add_geometry("mesh", mesh, material)
    elif style == "voxel":
        voxel = o3d.geometry.VoxelGrid().create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )
        render.scene.add_geometry("voxel", voxel, material)
    elif style == "lidar":
        points = np.concatenate((points, np.ones((len(fov_voxels), 1))), axis=1)
        points = (T_velo_2_cam @ points.T).T[:, :3]
        road_mask = fov_voxels[:, 3] >= 9
        num_thetas = 40
        num_phis = 200
        lidar_center = np.array([0, 0, -26])
        lidar_points = points - lidar_center
        lidar_points[:, 0] += np.random.normal(0, 0.05, len(lidar_points))
        lidar_points[:, 2] += np.random.normal(0, 0.05, len(lidar_points))
        thetas = np.arctan(
            lidar_points[:, 1]
            / np.sqrt(lidar_points[:, 0] ** 2 + lidar_points[:, 2] ** 2)
        )
        on_lidar = (
            np.abs(thetas * num_thetas - (thetas * num_thetas).round())
            < np.abs(thetas) / 4
        )
        thetas = (thetas * num_thetas).round() / num_thetas
        lidar_points[:, 1] = np.tan(thetas) * np.sqrt(
            lidar_points[:, 0] ** 2 + lidar_points[:, 2] ** 2
        )
        points[road_mask] = lidar_points[road_mask] + lidar_center

        pcd.points = o3d.utility.Vector3dVector(points[road_mask * on_lidar])
        pcd.colors = o3d.utility.Vector3dVector(colors[road_mask * on_lidar])
        voxels = o3d.geometry.PointCloud()
        voxels.points = o3d.utility.Vector3dVector(points[~road_mask])
        voxels.colors = o3d.utility.Vector3dVector(colors[~road_mask])
        voxels = o3d.geometry.VoxelGrid().create_from_point_cloud(
            voxels, voxel_size=voxel_size
        )
        render.scene.add_geometry("pcd", pcd, material)
        render.scene.add_geometry("voxels", voxels, material)

    # center = [0, 10, 10]
    # eye = [0, 0, -10]
    # up = [0, -1, 0]
    # render.setup_camera(vertical_fov, center, eye, up)

    intrinsics_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic_matrix = T_velo_2_cam
    render.setup_camera(intrinsics_matrix, extrinsic_matrix, width, height)
    img = render.render_to_image()
    o3d.io.write_image(dump_path, img, 9)


def dump(idx):
    pkl_name = pkl_names[idx]
    img_id = int(pkl_name.split(".")[0])
    img_name = pkl_name.replace(".pkl", "_left.png")
    pkl_path = os.path.join(pkl_dir, pkl_name)
    img_path = os.path.join(img_dir, img_name)

    # scan = config.file
    with open(pkl_path, "rb") as handle:
        b = pickle.load(handle)

    # for training and val
    y_pred = None
    if gt_view_mode:
        # for export_voxel tesing
        fov_mask_1 = b["fov_mask_1_1"]
        T_velo_2_cam = b["T_velo_2_cam"]
        y_pred = b["target_1_1"]
    else:
        fov_mask_1 = None
        if multi_view_mode:
            fov_mask_1 = b["fov_mask_1"][0, :, 0]
            for i in range(len(b["fov_mask_1"])):
                for j in range(len(b["fov_mask_1"][0, 0])):
                    if fov_mask_1 is None:
                        fov_mask_1 = b["fov_mask_1"][i, :, j]
                    else:
                        fov_mask_1 |= b["fov_mask_1"][i, :, j]
        else:
            fov_mask_1 = b["fov_mask_1"][0, :, 0]
            for i in range(1, len(b["fov_mask_1"][0, 0, :])):
                fov_mask_1 |= b["fov_mask_1"][0, :, i]
        T_velo_2_cam = b["T_velo_2_cam"][0]
        y_pred = b["y_pred"]
    vox_origin = b["vox_origin"]

    # Visualize
    dump_path = os.path.join(dump_dir, "{:06d}.png".format(idx))
    draw(
        y_pred,
        T_velo_2_cam,
        vox_origin,
        fov_mask_1,
        voxel_size=0.1,
        dump_path=dump_path,
        style=style,
    )
    vox_img = cv2.imread(dump_path)
    img = cv2.imread(img_path)
    new_w = img.shape[1]
    new_h = int(new_w / vox_img.shape[1] * vox_img.shape[0])
    vox_img = cv2.resize(vox_img, (new_w, new_h))
    dump_img = np.concatenate([img, vox_img], axis=0)
    cv2.imwrite(dump_path, dump_img)
    # print(f"Dump img: {dump_path}")


@hydra.main(config_path=None)
def main(config: DictConfig):
    global pkl_dir
    global img_dir
    global dump_dir
    global style
    global pkl_names
    global multi_view_mode
    global gt_view_mode

    if "gt_view_mode" in config:
        gt_view_mode = config.gt_view_mode

    if gt_view_mode:
        pkl_dir = "/data/dataset/TartanAir_full_preprocess/labels/office/Easy/P000/voxels_left/"
        img_dir = "/data/dataset/TartanAir_full/office/Easy/P000/image_left"
        dump_dir = "/data/test/"
    else:
        pkl_dir = "/data/workspace/OccDepth/output/tartanair/P005"
        img_dir = "/data/dataset/TartanAir_full/office/Easy/P005/image_left"
        dump_dir = "/data/workspace/OccDepth/output/tartanair/imgs_mesh"
    style = "mesh"
    multi_view_mode = False
    if "img_dir" in config:
        img_dir = config["img_dir"]
    if "pkl_dir" in config:
        pkl_dir = config["pkl_dir"]
    if "dump_dirs" in config:
        dump_dir = config["dump_dirs"]
    if "style" in config:
        style = config["style"]
    if "multi_view_mode" in config:
        multi_view_mode = config.multi_view_mode

    os.makedirs(dump_dir, exist_ok=True)

    pkl_names = os.listdir(pkl_dir)
    pkl_names = sorted(pkl_names)

    if style == "pcd2mesh":
        # create_from_point_cloud_ball_pivoting和多线程不兼容
        for idx in range(len(pkl_names)):
            dump(idx)
    else:
        pool = multiprocessing.Pool(processes=8)
        # pool.map(export_voxels_sub, depth_paths)
        r = list(
            tqdm(
                pool.imap(dump, range(len(pkl_names))),
                total=len(pkl_names),
                desc="dump_views",
            )
        )
        pool.close()
        pool.join()
    print("done")


if __name__ == "__main__":
    main()
