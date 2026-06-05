#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESDF 工具函数 / ESDF utility functions.

提供简单的 API，用于从深度图快速生成 ESDF，支持 CPU (SciPy) 和 GPU (CuPy) 两种计算方式。
Provide a simple API to quickly build an ESDF from a depth image, supporting both
CPU (SciPy) and GPU (CuPy) computation backends.
"""

import os
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import time

# 尝试导入 CuPy 用于 GPU 加速 / Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_edt
    CUPY_AVAILABLE = True
    print("[ESDF] ✅ CuPy 可用,支持 GPU 加速 EDT")
except ImportError:
    CUPY_AVAILABLE = False
    print("[ESDF] ⚠️ CuPy 不可用,将使用 CPU SciPy EDT")

def quick_depth_to_esdf(depth_image, camera_intrinsics,
                       voxel_size=0.05, grid_size=(80, 80, 40),
                       grid_origin=(0.0, -2.0, -1.0),
                       surface_threshold=None, downsample_factor=2,
                       depth_scale=1000.0, max_depth=8.0,
                       use_gpu=True, verbose=False):
    """
    快速从深度图生成 ESDF 的简化接口 / Simplified interface to build an ESDF from a depth image.

    参数 / Args:
        depth_image: 深度图数组 (H, W) 或文件路径字符串 / Depth image array (H, W) or a file path string.
        camera_intrinsics: 相机内参字典 {'fx', 'fy', 'ppx', 'ppy'} / Camera intrinsics dict {'fx', 'fy', 'ppx', 'ppy'}.
        voxel_size: 体素大小（米）/ Voxel size in meters.
        grid_size: 体素网格尺寸 (nx, ny, nz) / Voxel grid size (nx, ny, nz).
        grid_origin: 网格原点相对于相机的位置 (x, y, z) / Grid origin relative to the camera (x, y, z).
        surface_threshold: 表面厚度阈值（米）/ Surface thickness threshold in meters.
        downsample_factor: 点云下采样因子 / Point cloud downsampling factor.
        depth_scale: 深度缩放因子（PNG 图像通常为 1000.0）/ Depth scale factor (typically 1000.0 for PNG images).
        max_depth: 最大有效深度（米）/ Maximum valid depth in meters.
        use_gpu: 是否使用 GPU 加速（需要 CuPy）/ Whether to use GPU acceleration (requires CuPy).

    返回 / Returns:
        esdf: ESDF 数组 (nx, ny, nz) / ESDF array (nx, ny, nz).
        metadata: 包含网格信息的字典 / Dict containing grid metadata.
    """

    # 1. 加载深度图 / Load the depth image
    if isinstance(depth_image, str):
        # 从文件路径加载 / Load from a file path
        if depth_image.endswith('.png'):
            depth_array = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
            if depth_array is None:
                raise ValueError(f"无法读取深度图: {depth_image}")
            depth_array = depth_array.astype(np.float32) / depth_scale
        elif depth_image.endswith('.npy'):
            depth_array = np.load(depth_image).astype(np.float32)
        else:
            raise ValueError(f"不支持的文件格式: {depth_image}")
    else:
        # 直接使用数组 / Use the array directly
        depth_array = depth_image.astype(np.float32)

    # 裁剪深度值 / Clip depth values
    depth_array = np.clip(depth_array, 0, max_depth)
    depth_array[depth_array <= 0] = 0

    # 2. 深度图转点云 / Convert the depth image to a point cloud
    points = depth_to_pointcloud(depth_array, camera_intrinsics, downsample_factor)

    if len(points) == 0:
        print("警告: 点云为空，返回全正值ESDF")
        grid_size = np.array(grid_size)
        return np.full(grid_size, float('inf')), {}

    # 自动将表面阈值设为体素大小的一半（更合理的默认值）
    # Default the surface threshold to half the voxel size (a more sensible default)
    if surface_threshold is None:
        surface_threshold = voxel_size * 0.5

    # 3. 计算 ESDF / Compute the ESDF
    esdf = compute_esdf_from_pointcloud(
        points, voxel_size, grid_size, grid_origin, surface_threshold, use_gpu, verbose=verbose
    )

    # 4. 准备元数据 / Prepare the metadata
    grid_origin = np.array(grid_origin)
    grid_size = np.array(grid_size)
    min_bound = grid_origin
    max_bound = grid_origin + grid_size * voxel_size
    
    metadata = {
        'voxel_size': voxel_size,
        'grid_size': grid_size.tolist(),
        'grid_origin': grid_origin.tolist(),
        'grid_bounds': [min_bound.tolist(), max_bound.tolist()],
        'camera_intrinsics': camera_intrinsics,
        'esdf_range': [float(esdf.min()), float(esdf.max())],
        'occupied_voxels': int(np.sum(esdf < 0)),
        'total_voxels': int(np.prod(grid_size)),
        'point_count': len(points)
    }
    
    return esdf, metadata

def depth_to_pointcloud(depth_array, camera_intrinsics, downsample_factor=1):
    """
    将深度图转换为 3D 点云 / Convert a depth image into a 3D point cloud.

    参数 / Args:
        depth_array: 深度数组 (H, W)，单位米 / Depth array (H, W) in meters.
        camera_intrinsics: 相机内参字典 {'fx', 'fy', 'ppx', 'ppy'} / Camera intrinsics dict {'fx', 'fy', 'ppx', 'ppy'}.
        downsample_factor: 下采样因子 / Downsampling factor.

    返回 / Returns:
        points: 3D 点云数组 (N, 3) / 3D point cloud array (N, 3).
    """
    h, w = depth_array.shape
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    ppx = camera_intrinsics['ppx']
    ppy = camera_intrinsics['ppy']

    # 创建像素坐标网格 / Build the pixel coordinate grid
    u_coords = np.arange(0, w, downsample_factor)
    v_coords = np.arange(0, h, downsample_factor)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # 获取深度值 / Sample the depth values
    depth_sampled = depth_array[v_grid, u_grid]

    # 过滤有效深度 / Filter for valid depth
    valid_mask = depth_sampled > 0
    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    depth_valid = depth_sampled[valid_mask]

    # 像素坐标转世界坐标 / Convert pixel coordinates to world coordinates
    x = (u_valid - ppx) * depth_valid / fx
    y = (v_valid - ppy) * depth_valid / fy
    z = depth_valid
    
    return np.stack([x, y, z], axis=1)

def compute_esdf_from_pointcloud(points, voxel_size, grid_size, grid_origin, surface_threshold, use_gpu=True, verbose=True):
    """
    高效且准确地从点云直接计算 ESDF，支持 CPU 和 GPU 两种模式。
    Efficiently and accurately compute an ESDF directly from a point cloud,
    supporting both CPU and GPU backends.

    参数 / Args:
        points: 点云坐标 (N, 3) / Point cloud coordinates (N, 3).
        voxel_size: 体素大小 / Voxel size.
        grid_size: 网格尺寸 (nx, ny, nz) / Grid size (nx, ny, nz).
        grid_origin: 网格原点 (x, y, z) / Grid origin (x, y, z).
        surface_threshold: 表面厚度阈值 / Surface thickness threshold.
        use_gpu: 是否使用 GPU 加速（体素网格生成 + EDT）/ Whether to use GPU acceleration (voxel grid generation + EDT).

    返回 / Returns:
        esdf: ESDF 数组 (nx, ny, nz) / ESDF array (nx, ny, nz).
    """
    # 决定使用 GPU 还是 CPU / Decide whether to use GPU or CPU
    use_gpu_actual = use_gpu and CUPY_AVAILABLE
    backend = "GPU (CuPy)" if use_gpu_actual else "CPU (NumPy/SciPy)"
    if verbose: print(f"  使用 [{backend}] 生成占用网格...")
    start_time = time.time()

    # 确保输入为 numpy 数组 / Make sure the inputs are numpy arrays
    grid_size_np = np.array(grid_size, dtype=int)
    grid_origin_np = np.array(grid_origin, dtype=float)

    if len(points) == 0:
        if verbose: print("  警告：点云为空，返回全自由空间ESDF")
        return np.full(grid_size_np, voxel_size * np.max(grid_size_np), dtype=np.float32)

    if use_gpu_actual:
        # ==================== GPU 路径 / GPU path ====================
        # 转换到 GPU / Move data to the GPU
        points_gpu = cp.asarray(points)
        grid_origin_gpu = cp.asarray(grid_origin_np)

        # 计算体素索引 / Compute voxel indices
        point_indices_gpu = cp.floor((points_gpu - grid_origin_gpu) / voxel_size).astype(cp.int32)

        # 边界过滤 / Filter by grid bounds
        valid_mask_gpu = (
            (point_indices_gpu[:, 0] >= 0) & (point_indices_gpu[:, 0] < grid_size_np[0]) &
            (point_indices_gpu[:, 1] >= 0) & (point_indices_gpu[:, 1] < grid_size_np[1]) &
            (point_indices_gpu[:, 2] >= 0) & (point_indices_gpu[:, 2] < grid_size_np[2])
        )
        valid_indices_gpu = point_indices_gpu[valid_mask_gpu]
        
        num_valid = int(cp.sum(valid_mask_gpu))
        if verbose: print(f"    有效点云: {num_valid}/{len(points)} ({100*num_valid/len(points):.1f}%)")
        
        # 标记占用：跳过 unique()，直接赋值（bool 数组天然去重）
        # Mark occupancy: skip unique() and assign directly (a bool array deduplicates inherently)
        if num_valid > 0:
            occupancy_gpu = cp.zeros(grid_size_np, dtype=bool)
            occupancy_gpu[valid_indices_gpu[:, 0], valid_indices_gpu[:, 1], valid_indices_gpu[:, 2]] = True

            initial_occupied = int(cp.sum(occupancy_gpu))
            if verbose: print(f"    占用体素: {initial_occupied} 个")

            # 表面膨胀 (GPU) / Surface dilation (GPU)
            if surface_threshold > voxel_size * 0.8:
                from scipy.ndimage import generate_binary_structure
                dilation_radius = max(1, int(np.round(surface_threshold / voxel_size)))
                struct_elem = generate_binary_structure(3, 1)
                struct_elem_gpu = cp.asarray(struct_elem)
                
                from cupyx.scipy.ndimage import binary_dilation as cp_binary_dilation
                for _ in range(dilation_radius):
                    occupancy_gpu = cp_binary_dilation(occupancy_gpu, structure=struct_elem_gpu)
                
                final_occupied = int(cp.sum(occupancy_gpu))
                if verbose: print(f"    表面膨胀: {initial_occupied} -> {final_occupied} 体素 (膨胀半径: {dilation_radius})")
            
            # 转回 CPU 用于后续 ESDF 计算 / Move back to CPU for the subsequent ESDF computation
            occupancy = cp.asnumpy(occupancy_gpu)
        else:
            occupancy = np.zeros(grid_size_np, dtype=bool)

    else:
        # ==================== CPU 路径（原始实现）/ CPU path (original implementation) ====================
        occupancy = np.zeros(grid_size_np, dtype=bool)

        # 计算体素索引 / Compute voxel indices
        point_indices = np.floor((points - grid_origin_np) / voxel_size).astype(int)

        # 边界过滤 / Filter by grid bounds
        valid_mask = (
            (point_indices[:, 0] >= 0) & (point_indices[:, 0] < grid_size_np[0]) &
            (point_indices[:, 1] >= 0) & (point_indices[:, 1] < grid_size_np[1]) &
            (point_indices[:, 2] >= 0) & (point_indices[:, 2] < grid_size_np[2])
        )
        valid_indices = point_indices[valid_mask]
        if verbose: print(f"    有效点云: {len(valid_indices)}/{len(points)} ({100*len(valid_indices)/len(points):.1f}%)")
        
        # 标记占用：CPU 使用 unique 去重 / Mark occupancy: CPU uses unique() to deduplicate
        if len(valid_indices) > 0:
            unique_indices = np.unique(valid_indices, axis=0)
            occupancy[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = True
            if verbose: print(f"    占用体素: {len(unique_indices)} 个")

            # 表面膨胀 (CPU) / Surface dilation (CPU)
            initial_occupied = np.sum(occupancy)
            if surface_threshold > voxel_size * 0.8:
                from scipy.ndimage import binary_dilation, generate_binary_structure
                dilation_radius = max(1, int(np.round(surface_threshold / voxel_size)))
                struct_elem = generate_binary_structure(3, 1)
                
                for _ in range(dilation_radius):
                    occupancy = binary_dilation(occupancy, structure=struct_elem)
                
                final_occupied = np.sum(occupancy)
                if verbose: print(f"    表面膨胀: {initial_occupied} -> {final_occupied} 体素 (膨胀半径: {dilation_radius})")
    
    grid_time = time.time() - start_time
    if verbose: print(f"  占用网格生成完成 [{backend}]，耗时: {grid_time:.3f}秒")

    # 决定使用 GPU 还是 CPU 计算 ESDF / Decide whether to compute the ESDF on GPU or CPU
    use_gpu_actual = use_gpu and CUPY_AVAILABLE
    backend = "GPU (CuPy)" if use_gpu_actual else "CPU (SciPy)"
    if verbose: print(f"  计算ESDF距离场 [{backend}]...")
    start_time = time.time()

    # 检查占用情况 / Inspect the occupancy
    total_voxels = np.prod(grid_size)
    occupied_voxels = np.sum(occupancy)
    occupancy_ratio = occupied_voxels / total_voxels

    if occupied_voxels == 0:
        # 没有障碍物：全部为自由空间 / No obstacles: everything is free space
        if verbose: print("    无障碍物，生成全自由空间ESDF")
        esdf = np.full(grid_size, voxel_size * np.max(grid_size), dtype=np.float32)
    elif occupied_voxels == total_voxels:
        # 全是障碍物：全部位于障碍物内部 / All obstacles: everything is inside an obstacle
        if verbose: print("    全为障碍物，生成全负值ESDF")
        esdf = np.full(grid_size, -voxel_size * np.max(grid_size), dtype=np.float32)
    else:
        # 正常情况：自由空间与障碍物混合 / Normal case: a mix of free and occupied space
        sampling = [voxel_size, voxel_size, voxel_size]

        if use_gpu_actual:
            # GPU 路径：使用 CuPy / GPU path: use CuPy
            # 将 occupancy 传输到 GPU / Transfer occupancy to the GPU
            occupancy_gpu = cp.asarray(occupancy)

            # 在 GPU 上计算外部距离和内部距离 / Compute external and internal distances on the GPU
            external_dist_gpu = cp_edt(~occupancy_gpu, sampling=sampling)
            internal_dist_gpu = cp_edt(occupancy_gpu, sampling=sampling)

            # 组合 ESDF（在 GPU 上）/ Combine into the ESDF (on the GPU)
            esdf_gpu = cp.where(occupancy_gpu, -internal_dist_gpu, external_dist_gpu)

            # 传输回 CPU / Transfer back to the CPU
            esdf = cp.asnumpy(esdf_gpu).astype(np.float32)
        else:
            # CPU 路径：使用 SciPy（原始实现）/ CPU path: use SciPy (original implementation)
            # 外部距离：自由空间到最近障碍物的距离 / External distance: from free space to the nearest obstacle
            external_dist = distance_transform_edt(~occupancy, sampling=sampling)

            # 内部距离：障碍物内部到最近边界的距离 / Internal distance: from inside an obstacle to the nearest boundary
            internal_dist = distance_transform_edt(occupancy, sampling=sampling)

            # 组合 ESDF：障碍物内部为负值，自由空间为正值
            # Combine into the ESDF: negative inside obstacles, positive in free space
            esdf = np.where(occupancy, -internal_dist, external_dist).astype(np.float32)

    esdf_time = time.time() - start_time
    if verbose: print(f"  ESDF计算完成 [{backend}]，耗时: {esdf_time:.3f}秒")

    return esdf




def query_esdf_value(esdf, query_point, voxel_size, grid_origin):
    """
    查询 ESDF 中指定点的距离值 / Query the distance value at a given point in the ESDF.

    参数 / Args:
        esdf: ESDF 数组 (nx, ny, nz) / ESDF array (nx, ny, nz).
        query_point: 查询点世界坐标 (x, y, z) / World coordinates of the query point (x, y, z).
        voxel_size: 体素大小 / Voxel size.
        grid_origin: 网格原点 / Grid origin.

    返回 / Returns:
        distance: 距离值，如果超出范围返回 None / Distance value, or None if the point is out of bounds.
    """
    grid_origin = np.array(grid_origin)
    query_point = np.array(query_point)

    # 转换为体素索引 / Convert to a voxel index
    idx_f = (query_point - grid_origin) / voxel_size
    idx = np.round(idx_f).astype(int)

    # 检查边界 / Check the bounds
    if np.any(idx < 0) or np.any(idx >= esdf.shape):
        return None
    
    return float(esdf[idx[0], idx[1], idx[2]])

def create_simple_esdf_for_planning(depth_image_path, camera_intrinsics, 
                                   planning_range=3.0, resolution=0.02):
    """
    为路径规划创建简化的 ESDF / Create a simplified ESDF for path planning.

    参数 / Args:
        depth_image_path: 深度图路径 / Path to the depth image.
        camera_intrinsics: 相机内参 / Camera intrinsics.
        planning_range: 规划范围（米）/ Planning range in meters.
        resolution: 分辨率（米）/ Resolution in meters.

    返回 / Returns:
        esdf: ESDF 数组 / ESDF array.
        metadata: 元数据 / Metadata.
    """
    # 自动计算网格尺寸 / Automatically compute the grid size
    grid_size_voxels = int(planning_range / resolution)
    grid_size = (grid_size_voxels, grid_size_voxels, grid_size_voxels // 2)

    # 网格原点：相机前方，左右居中，底部适当下移
    # Grid origin: in front of the camera, centered laterally, shifted down at the bottom
    grid_origin = (0.0, -planning_range/2, -planning_range/4)

    return quick_depth_to_esdf(
        depth_image_path, camera_intrinsics,
        voxel_size=resolution,
        grid_size=grid_size,
        grid_origin=grid_origin,
        surface_threshold=resolution * 0.5,  # 表面厚度为分辨率的一半 / Surface thickness is half the resolution
        downsample_factor=max(1, int(0.01 / resolution))  # 根据分辨率调整下采样 / Adjust downsampling based on resolution
    )

# 使用示例 / Usage examples
if __name__ == "__main__":
    # 示例 1：基本使用 / Example 1: basic usage
    print("=== ESDF工具函数使用示例 ===")

    # 相机参数 / Camera parameters
    camera_intrinsics = {
        'fx': 389.551,
        'fy': 389.551,
        'ppx': 324.211,
        'ppy': 235.656
    }

    # 测试文件路径 / Test file path
    test_depth = os.environ.get("TEST_DEPTH", "real_test/stair/depth_image_mm.png")

    if os.path.exists(test_depth):
        print(f"\n1. 快速生成ESDF...")
        esdf, metadata = quick_depth_to_esdf(test_depth, camera_intrinsics)
        print(f"ESDF形状: {esdf.shape}")
        print(f"距离范围: [{esdf.min():.3f}, {esdf.max():.3f}]")

        print(f"\n2. 查询特定点的距离值...")
        # 查询相机前方 1 米处的距离 / Query the distance 1 meter in front of the camera
        dist = query_esdf_value(esdf, [1.0, 0.0, 0.0],
                               metadata['voxel_size'], metadata['grid_origin'])
        if dist is not None:
            print(f"点 (1.0, 0.0, 0.0) 的ESDF值: {dist:.3f}m")
        else:
            print("查询点超出ESDF范围")
        
        print(f"\n3. 为路径规划创建ESDF...")
        planning_esdf, planning_meta = create_simple_esdf_for_planning(
            test_depth, camera_intrinsics, planning_range=2.0, resolution=0.05
        )
        print(f"路径规划ESDF形状: {planning_esdf.shape}")
        
    else:
        print(f"测试文件不存在: {test_depth}")
        print("请提供有效的深度图路径进行测试")
