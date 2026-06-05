#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVBlox ESDF 实现 / NVBlox ESDF implementation.

使用 NVIDIA nvblox_torch 库进行 GPU 加速的 ESDF 生成和查询。
Use the NVIDIA nvblox_torch library for GPU-accelerated ESDF generation and queries.

参考 / Reference: https://nvidia-isaac.github.io/nvblox/pages/torch_examples_esdf.html
"""

import torch
import numpy as np
from nvblox_torch.mapper import Mapper, MapperParams, QueryType
from nvblox_torch.constants import constants
import time
import os
import logging
from typing import Dict, Tuple, Optional, Union

# 抑制 nvblox 的 C++ 日志输出（INFO 级别），这些日志来自 Google logging (glog)。
# Suppress nvblox C++ log output (INFO level); these logs come from Google logging (glog).
os.environ.setdefault('GLOG_minloglevel', '1')  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL


class NvbloxESDFMapper:
    """
    基于 nvblox 的 GPU 加速 ESDF 映射器 / nvblox-based GPU-accelerated ESDF mapper.

    提供与原始 esdf_utils.py 兼容的接口。
    Provides an interface compatible with the original esdf_utils.py.
    """
    
    def __init__(self, 
                 voxel_size: float = 0.1,
                 device: str = 'cuda',
                 verbose: bool = False):
        """
        初始化 nvblox ESDF 映射器 / Initialize the nvblox ESDF mapper.

        参数 / Args:
            voxel_size: 体素大小（米） / Voxel size in meters.
            device: 计算设备（'cuda' 或 'cpu'） / Compute device ('cuda' or 'cpu').
            verbose: 是否打印详细信息 / Whether to print verbose information.
        """
        self.voxel_size = voxel_size
        self.device = device
        self.verbose = verbose
        
        # 创建 mapper 参数 / Create mapper parameters
        params = MapperParams()

        # 创建 nvblox mapper（TSDF integrator） / Create the nvblox mapper (TSDF integrator)
        self.mapper = Mapper(
            voxel_sizes_m=voxel_size,
            mapper_parameters=params
        )
        
        if verbose:
            print(f"✅ NvbloxESDFMapper 初始化完成")
            print(f"   - 体素大小: {voxel_size}m")
            print(f"   - 计算设备: {device}")
    
    def integrate_depth(self,
                       depth_image: Union[np.ndarray, torch.Tensor],
                       camera_intrinsics: Dict[str, float],
                       camera_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       max_depth: float = 8.0,
                       truncation_distance: float = 0.4) -> None:
        """
        将深度图整合到 TSDF 中 / Integrate a depth image into the TSDF.

        参数 / Args:
            depth_image: 深度图 (H, W)，单位米 / Depth image (H, W) in meters.
            camera_intrinsics: 相机内参 {'fx', 'fy', 'ppx', 'ppy'} / Camera intrinsics {'fx', 'fy', 'ppx', 'ppy'}.
            camera_pose: 相机位姿（4x4 变换矩阵），默认为原点 / Camera pose (4x4 transform), defaults to origin.
            max_depth: 最大有效深度（米） / Maximum valid depth in meters.
            truncation_distance: TSDF 截断距离（米） / TSDF truncation distance in meters.
        """
        start_time = time.time()
        
        # 1. 转换深度图为 torch tensor / Convert the depth image to a torch tensor
        if isinstance(depth_image, np.ndarray):
            depth_tensor = torch.from_numpy(depth_image).float()
        else:
            depth_tensor = depth_image.float()

        # 确保在 CUDA 上 / Make sure the tensor lives on CUDA
        depth_tensor = depth_tensor.cuda()

        # 2. 构造相机内参 / Build the camera intrinsics
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics.get('ppx', camera_intrinsics.get('cx', depth_tensor.shape[1] / 2))
        cy = camera_intrinsics.get('ppy', camera_intrinsics.get('cy', depth_tensor.shape[0] / 2))
        
        height, width = depth_tensor.shape

        # 3. 构造相机位姿（默认为原点，Z 轴朝前） / Build the camera pose (defaults to origin, +Z forward)
        if camera_pose is None:
            # 默认位姿: 相机在原点，看向 +Z 方向 / Default pose: camera at origin looking toward +Z
            T_L_C = torch.eye(4)  # CPU 张量 / CPU tensor
        else:
            if isinstance(camera_pose, np.ndarray):
                T_L_C = torch.from_numpy(camera_pose).float()
            else:
                T_L_C = camera_pose.float().cpu()
        
        # 4. 使用 nvblox 的 add_depth_frame 接口 / Use the nvblox add_depth_frame interface.
        # API: add_depth_frame(depth_frame (H,W), t_w_c (4,4), intrinsics (3,3))
        # 注意: depth 在 GPU，pose 和 intrinsics 在 CPU。
        # Note: depth lives on GPU, while pose and intrinsics live on CPU.

        # 构造 3x3 内参矩阵 / Build the 3x3 intrinsics matrix
        intrinsics_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=torch.float32)  # CPU 张量 / CPU tensor
        
        self.mapper.add_depth_frame(
            depth_frame=depth_tensor,  # (H, W) CUDA 张量 / (H, W) CUDA tensor
            t_w_c=T_L_C,  # (4, 4) CPU 张量 / (4, 4) CPU tensor
            intrinsics=intrinsics_matrix  # (3, 3) CPU 张量 / (3, 3) CPU tensor
        )
        
        if self.verbose:
            elapsed = (time.time() - start_time) * 1000
            print(f"⚡ TSDF整合完成: {elapsed:.2f}ms")

    def update_esdf(self, max_distance: float = 4.0) -> None:
        """
        从 TSDF 更新 ESDF 层 / Update the ESDF layer from the TSDF.

        参数 / Args:
            max_distance: ESDF 最大计算距离（米） / Maximum ESDF computation distance in meters.
        """
        start_time = time.time()

        # nvblox 的 update_esdf 不需要参数 / nvblox update_esdf takes no arguments
        self.mapper.update_esdf()

        if self.verbose:
            elapsed = (time.time() - start_time) * 1000
            print(f"⚡ ESDF更新完成: {elapsed:.2f}ms")
    
    def query_esdf_distance(self, point_3d: Union[np.ndarray, torch.Tensor]) -> float:
        """
        查询单个 3D 点的 ESDF 距离 / Query the ESDF distance for a single 3D point.

        参数 / Args:
            point_3d: 3D 坐标 [x, y, z] / 3D coordinate [x, y, z].

        返回 / Returns:
            distance: ESDF 距离值（米），负值表示在障碍物内 / ESDF distance in meters; negative means inside an obstacle.
        """
        # 转换为 tensor / Convert to a tensor
        if isinstance(point_3d, np.ndarray):
            point_tensor = torch.from_numpy(point_3d).float()
        else:
            point_tensor = point_3d.float()

        point_tensor = point_tensor.cuda()

        # 确保形状为 (1, 3) / Ensure the shape is (1, 3)
        if point_tensor.dim() == 1:
            point_tensor = point_tensor.unsqueeze(0)

        # 查询 ESDF（使用 query_differentiable_layer） / Query the ESDF (via query_differentiable_layer)
        distance = self.mapper.query_differentiable_layer(
            QueryType.ESDF,
            point_tensor
        )

        # 处理未知值 / Handle unknown values
        unknown_dist = constants.esdf_unknown_distance
        if distance[0] == unknown_dist:
            return float('inf')  # 未知区域返回无穷大 / Return infinity for unknown regions

        return float(distance[0])
    
    def query_esdf_batch(self, points_3d: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        批量查询多个 3D 点的 ESDF 距离（高效） / Efficiently query ESDF distances for many 3D points.

        参数 / Args:
            points_3d: 3D 坐标数组 (N, 3) / Array of 3D coordinates (N, 3).

        返回 / Returns:
            distances: ESDF 距离数组 (N,) / Array of ESDF distances (N,).
        """
        # 转换为 tensor / Convert to a tensor
        if isinstance(points_3d, np.ndarray):
            points_tensor = torch.from_numpy(points_3d).float()
        else:
            points_tensor = points_3d.float()

        points_tensor = points_tensor.cuda()

        # 批量查询 / Batched query
        distances = self.mapper.query_differentiable_layer(
            QueryType.ESDF,
            points_tensor
        )

        # 处理未知值 / Handle unknown values
        unknown_dist = constants.esdf_unknown_distance
        distances[distances == unknown_dist] = float('inf')

        return distances.cpu().numpy()
    
    def get_tsdf_layer(self):
        """获取 TSDF 层视图 / Get a view of the TSDF layer."""
        return self.mapper.tsdf_layer_view()

    def clear(self) -> None:
        """清空所有体素数据 / Clear all voxel data."""
        self.mapper.clear()
        if self.verbose:
            print("🗑️  Mapper已清空")
    
    def get_statistics(self) -> Dict:
        """
        获取映射器统计信息 / Get mapper statistics.

        返回 / Returns:
            stats: 统计信息字典 / Dictionary of statistics.
        """
        return {
            'voxel_size': self.voxel_size,
            'device': self.device,
        }


def nvblox_depth_to_esdf(depth_image: Union[np.ndarray, str],
                         camera_intrinsics: Dict[str, float],
                         voxel_size: float = 0.1,
                         grid_size: Tuple[int, int, int] = (50, 30, 45),
                         grid_origin: Tuple[float, float, float] = (-2.5, -0.5, 0.0),
                         max_depth: float = 8.0,
                         device: str = 'cuda',
                         camera_pose: Optional[np.ndarray] = None,
                         verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    使用 nvblox 从深度图快速生成 ESDF / Quickly generate an ESDF from a depth image using nvblox.

    兼容原始 esdf_utils.quick_depth_to_esdf 的接口。
    Compatible with the original esdf_utils.quick_depth_to_esdf interface.

    参数 / Args:
        depth_image: 深度图数组 (H, W) 或文件路径 / Depth image array (H, W) or a file path.
        camera_intrinsics: 相机内参 {'fx', 'fy', 'ppx', 'ppy'} / Camera intrinsics {'fx', 'fy', 'ppx', 'ppy'}.
        voxel_size: 体素大小（米） / Voxel size in meters.
        grid_size: 体素网格尺寸 (nx, ny, nz) / Voxel grid size (nx, ny, nz).
        grid_origin: 网格原点 (x, y, z) / Grid origin (x, y, z).
        max_depth: 最大有效深度（米） / Maximum valid depth in meters.
        device: 计算设备 / Compute device.
        camera_pose: 相机位姿（4x4 矩阵） / Camera pose (4x4 matrix).
        verbose: 是否打印详细信息 / Whether to print verbose information.

    返回 / Returns:
        esdf: ESDF 数组 (nx, ny, nz) / ESDF array (nx, ny, nz).
        metadata: 包含网格信息的字典 / Dictionary containing grid information.
    """
    total_start = time.time()
    
    # 1. 加载深度图 / Load the depth image
    if isinstance(depth_image, str):
        import cv2
        if depth_image.endswith('.png'):
            depth_array = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
            if depth_array is None:
                raise ValueError(f"无法读取深度图: {depth_image}")
            depth_array = depth_array.astype(np.float32) / 1000.0  # 假设 PNG 为毫米 / Assume PNG is in millimeters
        elif depth_image.endswith('.npy'):
            depth_array = np.load(depth_image).astype(np.float32)
        else:
            raise ValueError(f"不支持的文件格式: {depth_image}")
    else:
        depth_array = depth_image.astype(np.float32)

    # 裁剪深度值 / Clip the depth values
    depth_array = np.clip(depth_array, 0, max_depth)
    
    if verbose:
        print(f"📊 深度图: {depth_array.shape}, 范围: [{depth_array.min():.2f}, {depth_array.max():.2f}]m")
    
    # 2. 创建 nvblox 映射器 / Create the nvblox mapper
    mapper = NvbloxESDFMapper(
        voxel_size=voxel_size,
        device=device,
        verbose=verbose
    )

    # 3. 整合深度图 / Integrate the depth image
    mapper.integrate_depth(
        depth_image=depth_array,
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose,
        max_depth=max_depth,
        truncation_distance=voxel_size * 4  # 常用值 / Common value
    )

    # 4. 更新 ESDF / Update the ESDF
    mapper.update_esdf(max_distance=max(grid_size) * voxel_size / 2)

    # 5. 提取 ESDF 网格到 numpy（在指定区域采样） / Extract the ESDF grid to numpy (sample over the given region)
    esdf_grid = _sample_esdf_grid(
        mapper=mapper,
        grid_size=grid_size,
        grid_origin=grid_origin,
        voxel_size=voxel_size
    )
    
    # 6. 准备元数据 / Prepare the metadata
    grid_origin_arr = np.array(grid_origin)
    grid_size_arr = np.array(grid_size)
    min_bound = grid_origin_arr
    max_bound = grid_origin_arr + grid_size_arr * voxel_size
    
    total_time = (time.time() - total_start) * 1000
    
    metadata = {
        'voxel_size': voxel_size,
        'grid_size': grid_size_arr.tolist(),
        'grid_origin': grid_origin_arr.tolist(),
        'grid_bounds': [min_bound.tolist(), max_bound.tolist()],
        'camera_intrinsics': camera_intrinsics,
        'esdf_range': [float(esdf_grid.min()), float(esdf_grid.max())],
        'occupied_voxels': int(np.sum(esdf_grid < 0)),
        'total_voxels': int(np.prod(grid_size)),
        'backend': 'nvblox_torch',
        'device': device,
        'total_time_ms': total_time
    }
    
    if verbose:
        print(f"✅ ESDF生成完成: {total_time:.2f}ms")
        print(f"   - 体素数: {metadata['total_voxels']}")
        print(f"   - 占用体素: {metadata['occupied_voxels']}")
        print(f"   - ESDF范围: [{metadata['esdf_range'][0]:.2f}, {metadata['esdf_range'][1]:.2f}]m")
    
    return esdf_grid, metadata


def _sample_esdf_grid(mapper: NvbloxESDFMapper,
                     grid_size: Tuple[int, int, int],
                     grid_origin: Tuple[float, float, float],
                     voxel_size: float) -> np.ndarray:
    """
    从 nvblox 稀疏 ESDF 中采样规则网格 / Sample a regular grid from the sparse nvblox ESDF.

    参数 / Args:
        mapper: NvbloxESDFMapper 实例 / A NvbloxESDFMapper instance.
        grid_size: 目标网格尺寸 (nx, ny, nz) / Target grid size (nx, ny, nz).
        grid_origin: 网格原点 (x, y, z) / Grid origin (x, y, z).
        voxel_size: 体素大小 / Voxel size.

    返回 / Returns:
        esdf_grid: 规则 ESDF 网格 (nx, ny, nz) / Regular ESDF grid (nx, ny, nz).
    """
    nx, ny, nz = grid_size
    ox, oy, oz = grid_origin

    # 生成采样点坐标（体素中心） / Generate sampling-point coordinates (voxel centers)
    x_coords = np.arange(nx) * voxel_size + ox + voxel_size / 2
    y_coords = np.arange(ny) * voxel_size + oy + voxel_size / 2
    z_coords = np.arange(nz) * voxel_size + oz + voxel_size / 2

    # 创建 3D 网格 / Create the 3D grid
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # 展平为点列表 (N, 3) / Flatten into a point list (N, 3)
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # 批量查询 ESDF / Batched ESDF query
    distances = mapper.query_esdf_batch(points)

    # 重塑为网格 / Reshape back into a grid
    esdf_grid = distances.reshape(nx, ny, nz)

    return esdf_grid


def create_esdf_query_function(mapper: NvbloxESDFMapper):
    """
    创建一个 ESDF 查询函数（用于 trajectory_evaluation.py） / Create an ESDF query function (used by trajectory_evaluation.py).

    参数 / Args:
        mapper: NvbloxESDFMapper 实例 / A NvbloxESDFMapper instance.

    返回 / Returns:
        query_fn: 查询函数 query_fn(point_3d) -> distance / Query function query_fn(point_3d) -> distance.
    """
    def query_fn(point_3d: np.ndarray) -> float:
        """查询 3D 点的 ESDF 距离 / Query the ESDF distance of a 3D point."""
        return mapper.query_esdf_distance(point_3d)

    # 添加批量查询方法 / Attach a batched-query method
    query_fn.batch = lambda points: mapper.query_esdf_batch(points)
    
    return query_fn


# ============================================================================
# 测试和示例代码 / Tests and example code
# ============================================================================

def test_nvblox_esdf():
    """测试 nvblox ESDF 功能 / Test the nvblox ESDF functionality."""
    print("=" * 60)
    print("测试 NVBlox ESDF 实现")
    print("=" * 60)

    # 1. 创建测试深度图 / Create a test depth image
    print("\n1️⃣  创建测试深度图...")
    height, width = 168, 224
    depth = np.ones((height, width), dtype=np.float32) * 3.0  # 3 米远的墙 / A wall 3 meters away

    # 添加一些障碍物 / Add some obstacles
    depth[60:100, 80:120] = 1.5  # 近处障碍物 / Near obstacle
    depth[120:150, 150:180] = 2.0  # 另一个障碍物 / Another obstacle
    
    print(f"   深度图形状: {depth.shape}")
    print(f"   深度范围: [{depth.min():.2f}, {depth.max():.2f}]m")
    
    # 2. 相机内参 / Camera intrinsics
    camera_intrinsics = {
        'fx': 389.551,
        'fy': 389.551,
        'ppx': 112.0,  # width / 2
        'ppy': 84.0    # height / 2
    }

    # 3. 测试 nvblox 实现 / Test the nvblox implementation
    print("\n2️⃣  使用NVBlox生成ESDF...")
    esdf, metadata = nvblox_depth_to_esdf(
        depth_image=depth,
        camera_intrinsics=camera_intrinsics,
        voxel_size=0.1,
        grid_size=(50, 30, 45),
        grid_origin=(-2.5, -1.5, 0.0),
        max_depth=8.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    print(f"\n✅ ESDF生成成功!")
    print(f"   - 网格尺寸: {esdf.shape}")
    print(f"   - ESDF范围: [{esdf.min():.3f}, {esdf.max():.3f}]m")
    print(f"   - 占用体素: {np.sum(esdf < 0)} / {esdf.size}")
    print(f"   - 总耗时: {metadata['total_time_ms']:.2f}ms")
    
    # 4. 测试点查询 / Test point queries
    print("\n3️⃣  测试ESDF查询...")
    mapper = NvbloxESDFMapper(
        voxel_size=0.1, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False
    )
    mapper.integrate_depth(depth, camera_intrinsics)
    mapper.update_esdf()
    
    test_points = np.array([
        [0.0, 0.0, 1.0],   # 正前方 1 米 / 1 m straight ahead
        [0.5, 0.0, 2.0],   # 右前方 2 米 / 2 m ahead-right
        [0.0, 0.0, 5.0],   # 正前方 5 米 / 5 m straight ahead
    ])
    
    for i, point in enumerate(test_points):
        distance = mapper.query_esdf_distance(point)
        print(f"   点 {i+1} {point}: {distance:.3f}m")
    
    # 5. 批量查询测试 / Batched-query test
    print("\n4️⃣  测试批量查询...")
    num_points = 1000
    random_points = np.random.randn(num_points, 3) * 2  # 随机点 / Random points
    
    start = time.time()
    distances = mapper.query_esdf_batch(random_points)
    elapsed = (time.time() - start) * 1000
    
    print(f"   查询 {num_points} 个点: {elapsed:.2f}ms ({num_points/elapsed*1000:.0f} 点/秒)")
    print(f"   距离范围: [{distances.min():.3f}, {distances.max():.3f}]m")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 检查 nvblox_torch 是否可用 / Check whether nvblox_torch is available
    try:
        import nvblox_torch
        print(f"✅ nvblox_torch 版本: {nvblox_torch.__version__}")
    except ImportError:
        print("❌ nvblox_torch 未安装!")
        print("   安装命令:")
        print("   pip install nvblox_torch")
        exit(1)
    
    # 检查 CUDA / Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA 不可用，将使用CPU (速度较慢)")

    print()

    # 运行测试 / Run the tests
    test_nvblox_esdf()
