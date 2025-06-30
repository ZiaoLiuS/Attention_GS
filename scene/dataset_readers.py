#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    attention_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        attention_path = ""
        # attention_path = os.path.join(os.path.dirname(images_folder),"complexity/", extr.name)
   
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,attention_path=attention_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def calculate_scale(points):
    """计算点云的尺度（边界框对角线长度）"""
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    return np.linalg.norm(max_vals - min_vals)

def add_scaled_gaussian_noise(points, intensity=0.1, mu=0):
    scale = calculate_scale(points)
    sigma = scale * intensity
    np.random.seed(11)
    noise = np.random.normal(scale * mu, sigma, points.shape)
    return points + noise


def fetchPly_noise(path, add_noise=True, noise_intensity=0.01, noise_mu=0.01):
    """
    读取PLY文件并可选地添加比例高斯噪声
    
    参数:
        path: PLY文件路径
        add_noise: 是否添加噪声(默认False)
        noise_intensity: 噪声强度(0-1)，相对于点云尺度(默认0.01)
        noise_mu: 噪声均值(默认0)
    
    返回:
        BasicPointCloud对象(包含原始或加噪的点云)
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # 提取原始数据
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    # 如果需要添加噪声
    if add_noise:
        positions = add_scaled_gaussian_noise(positions, 
                                           intensity=noise_intensity, 
                                           mu=noise_mu)
        print(f"Added Gaussian noise with intensity: {noise_intensity}")
        print(f"Actual noise sigma: {calculate_scale(positions) * noise_intensity:.6f}")
    
    return BasicPointCloud(points=positions, colors=np.random.random((len(positions), 3)), normals=np.zeros((len(positions), 3)))

def add_noise_points(original_points, original_colors, intensity=0.01, noise_ratio=0.1, mu=0):
    """
    在原始点附近添加新的噪声点（不改变原始点）
    
    参数:
        original_points: 原始点云位置(N×3)
        original_colors: 原始点云颜色(N×3)
        intensity: 噪声强度(0-1)，相对于点云尺度
        noise_ratio: 噪声点数量占原始点的比例(默认0.1，即10%)
        mu: 噪声均值(默认0)
    
    返回:
        combined_points: 合并后的点云位置
        combined_colors: 合并后的点云颜色
    """
    scale = calculate_scale(original_points)
    sigma = scale * intensity
    
    # 确定要添加的噪声点数量
    num_original = original_points.shape[0]
    num_noise = int(num_original * noise_ratio)
    
    # 随机选择原始点作为噪声点的基础
    base_indices = np.random.choice(num_original, num_noise, replace=True)
    base_points = original_points[base_indices]
    base_colors = original_colors[base_indices]
    
    # 生成噪声偏移量
    noise_offset = np.random.normal(mu, sigma, size=(num_noise, 3))
    
    # 创建噪声点
    noise_points = base_points + noise_offset
    noise_colors = base_colors  # 使用与基础点相同的颜色
    
    # 合并原始点和噪声点
    combined_points = np.vstack([original_points, noise_points])
    combined_colors = np.vstack([original_colors, noise_colors])
    
    return combined_points, combined_colors

def fetchPly_add_noise(path, add_noise=False, noise_intensity=0.02, noise_ratio=0.5, noise_mu=0):
    """
    读取PLY文件并可选地在原始点附近添加新噪声点
    
    参数:
        path: PLY文件路径
        add_noise: 是否添加噪声点(默认False)
        noise_intensity: 噪声强度(0-1)，相对于点云尺度(默认0.01)
        noise_ratio: 噪声点数量占原始点的比例(默认0.1)
        noise_mu: 噪声均值(默认0)
    
    返回:
        BasicPointCloud对象(包含原始点或原始+噪声点)
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # 提取原始数据
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    # 如果需要添加噪声点
    if add_noise:
        positions, colors = add_noise_points(
            positions, colors,
            intensity=noise_intensity,
            noise_ratio=noise_ratio,
            mu=noise_mu
        )
        # 注意：噪声点没有法线信息，这里简单复制原始法线或使用零值
        # 更复杂的实现可以根据噪声偏移方向计算新法线
        if normals is not None:
            base_normals = normals[np.random.choice(len(normals), len(positions)-len(normals))]
            normals = np.vstack([normals, base_normals])
        
        print(f"Added {int(len(positions)-len(normals))} noise points ({noise_ratio*100}% of original)")
        print(f"Noise intensity: {intensity} (sigma: {calculate_scale(positions)*noise_intensity:.6f})")
        
    print(colors)
    
    return BasicPointCloud(points=positions, colors=np.random.random((len(positions), 3)), normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, ours=None, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

            
    if ours is not None:
        cam_pos = []
        for k in cam_extrinsics.keys():
            cam_pos.append(cam_extrinsics[k].tvec)
        cam_pos = np.array(cam_pos)
        min_cam_pos = np.min(cam_pos)
        max_cam_pos = np.max(cam_pos)
        mean_cam_pos = (min_cam_pos + max_cam_pos) / 2.0
        cube_mean = (max_cam_pos - min_cam_pos) * 1.5
            
    if ours is not None and ours == 0:
        num_pts = 100000
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 - nerf_normalization["radius"] * 1.5
        xyz = xyz + nerf_normalization["translate"]
        print(f"Generating random point cloud ({num_pts})...")
    
    if ours is not None and ours == 1:
        num_pts = 10
        xyz = np.random.random((num_pts, 3)) * (max_cam_pos - min_cam_pos) * 3 - (cube_mean - mean_cam_pos)
        print(f"Generating OUR point cloud ({num_pts})...")
    
    if ours is not None and ours == 2:
        num_sclaes = 1
        base_num = 4
        xyz = []
        for i in range(num_sclaes):
            num_pts = base_num * 8**i
            print(f"Generating scale {i} point cloud ({num_pts})...")
            a = np.random.random((num_pts, 3)) * (max_cam_pos - min_cam_pos) * 3 - (cube_mean - mean_cam_pos)
            xyz.append(a)
        xyz = np.concatenate(xyz)
        print(f"Generating yifeng point cloud ({len(xyz)})...")
        num_pts = len(xyz)

    if ours is not None:
        shs = np.random.random((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
        # storePly(ply_path, xyz, SH2RGB(shs) * 255)

    # storePly(os.path.join(path, "sparse/0/points3D_noise.ply"), pcd.points, pcd.colors)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            attention_path = os.path.join(os.path.dirname(images_folder),"complexity/", extr.name)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path,attention_path=attention_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}