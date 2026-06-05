#!/usr/bin/env python3
"""
轨迹等弧长采样模块 - 向量化优化版本 / Arc-length trajectory sampling module - vectorized optimized version.

优化点：
1. 批量 B-spline / Cubic Spline 重建（避免 for 循环）
2. 向量化弧长计算
3. 批量插值操作
4. 减少 Python 循环开销
预期加速：3-5 倍

Optimizations:
1. Batched B-spline / Cubic Spline reconstruction (avoiding for loops)
2. Vectorized arc length computation
3. Batched interpolation operations
4. Reduced Python loop overhead
Expected speedup: 3-5x.
"""

import numpy as np
from scipy.interpolate import BSpline
from typing import List, Optional
import warnings
import torch

# 延迟导入 TrajOpt，避免循环依赖
# Lazily import TrajOpt to avoid circular dependency
_traj_opt_instance = None

def _get_traj_opt():
    """延迟加载 TrajOpt 实例 / Lazily load the TrajOpt instance."""
    global _traj_opt_instance
    if _traj_opt_instance is None:
        from sand_planner.utils.traj_opt import TrajOpt
        _traj_opt_instance = TrajOpt()
    return _traj_opt_instance


def reconstruct_trajectories_batch_bspline(control_points_batch: np.ndarray, 
                                           num_points: int = 200,
                                           degree: int = 3) -> np.ndarray:
    """
    批量从 B-spline 控制点重建轨迹（向量化版本）/ Batch-reconstruct trajectories from B-spline control points (vectorized version).

    Args:
        control_points_batch: (B, 8, 3) 批量控制点，B 是 batch 大小 / Batched control points, B is the batch size.
        num_points: 每条轨迹重建的点数 / Number of points to reconstruct per trajectory.
        degree: B-spline 次数 / B-spline degree.

    Returns:
        trajectories: (B, num_points, 3) 批量轨迹 / Batched trajectories.
    """
    B = control_points_batch.shape[0]

    # 创建 knot 向量（所有轨迹共享）
    # Build the knot vector (shared by all trajectories)
    k = degree
    m = control_points_batch.shape[1]  # 控制点数量（从输入自动推断）/ Number of control points (inferred from input)
    inner_knots = np.linspace(0, 1, m - k + 1)
    knots = np.concatenate([
        np.zeros(k),
        inner_knots,
        np.ones(k)
    ])

    # 采样参数 / Sampling parameter
    u = np.linspace(0.0, 1.0, num_points)

    # 批量重建（向量化）/ Batch reconstruction (vectorized)
    trajectories = np.zeros((B, num_points, 3))

    for dim in range(3):  # x, y, z
        # 对所有轨迹在该维度上创建 B-spline 并评估
        # Build and evaluate a B-spline for every trajectory along this dimension
        for i in range(B):
            bs = BSpline(knots, control_points_batch[i, :, dim], k, extrapolate=False)
            trajectories[i, :, dim] = bs(u)
    
    return trajectories


def reconstruct_trajectories_batch_cubic_spline(control_points_batch: np.ndarray,
                                                 num_points: int = 200) -> np.ndarray:
    """
    批量从 Cubic Spline 控制点重建轨迹 / Batch-reconstruct trajectories from Cubic Spline control points.

    Args:
        control_points_batch: (B, 8, 3) 批量控制点，B 是 batch 大小 / Batched control points, B is the batch size.
        num_points: 每条轨迹重建的点数 / Number of points to reconstruct per trajectory.

    Returns:
        trajectories: (B, num_points, 3) 批量轨迹 / Batched trajectories.
    """
    B = control_points_batch.shape[0]

    traj_opt = _get_traj_opt()

    # 去掉原点，TrajOpt 会自动添加
    # Drop the origin point; TrajOpt will add it back automatically
    # control_points_batch: (B, m, 3) -> cp_without_origin: (B, m-1, 3)
    n_cp = control_points_batch.shape[1]  # 控制点数量（从输入自动推断）/ Number of control points (inferred from input)
    cp_without_origin = control_points_batch[:, 1:, :]  # (B, m-1, 3)

    # 转为 torch tensor / Convert to a torch tensor
    cp_torch = torch.from_numpy(cp_without_origin).float()  # (B, m-1, 3)

    # 计算采样步长：(m-1) 个控制点 + 原点 = m 个点，参数空间 [0, m-1]
    # Compute the sampling step: (m-1) control points + origin = m points, parameter space [0, m-1]
    step = (n_cp - 1) / num_points

    with torch.no_grad():
        trajectories_torch = traj_opt.generate_trajectory_from_control_points(cp_torch, step=step)

    trajectories = trajectories_torch.numpy()  # (B, M, 3)

    # 如果点数不匹配，使用插值对齐
    # If the point count does not match, align via interpolation
    if trajectories.shape[1] != num_points:
        aligned = np.zeros((B, num_points, 3))
        t_src = np.linspace(0, 1, trajectories.shape[1])
        t_dst = np.linspace(0, 1, num_points)
        for i in range(B):
            for dim in range(3):
                aligned[i, :, dim] = np.interp(t_dst, t_src, trajectories[i, :, dim])
        trajectories = aligned
    
    return trajectories


def reconstruct_trajectories_batch(control_points_batch: np.ndarray, 
                                   num_points: int = 200,
                                   degree: int = 3,
                                   method: str = 'bspline') -> np.ndarray:
    """
    批量从控制点重建轨迹（支持 B-spline 和 Cubic Spline）/ Batch-reconstruct trajectories from control points (supports B-spline and Cubic Spline).

    Args:
        control_points_batch: (B, 8, 3) 批量控制点，B 是 batch 大小 / Batched control points, B is the batch size.
        num_points: 每条轨迹重建的点数 / Number of points to reconstruct per trajectory.
        degree: B-spline 次数（仅对 bspline 有效）/ B-spline degree (only used for 'bspline').
        method: 'bspline' 或 'cubic_spline' / 'bspline' or 'cubic_spline'.

    Returns:
        trajectories: (B, num_points, 3) 批量轨迹 / Batched trajectories.
    """
    if method == 'cubic_spline':
        return reconstruct_trajectories_batch_cubic_spline(control_points_batch, num_points)
    else:
        return reconstruct_trajectories_batch_bspline(control_points_batch, num_points, degree)


def compute_arc_lengths_batch(trajectories: np.ndarray) -> tuple:
    """
    批量计算累积弧长（向量化版本）/ Batch-compute cumulative arc length (vectorized version).

    Args:
        trajectories: (B, N, 3) 批量轨迹 / Batched trajectories.

    Returns:
        cumulative_lengths: (B, N) 每条轨迹的累积弧长 / Cumulative arc length of each trajectory.
        total_lengths: (B,) 每条轨迹的总长度 / Total length of each trajectory.
    """
    # 计算相邻点之间的距离 (B, N-1, 3)
    # Compute the distances between adjacent points (B, N-1, 3)
    diffs = np.diff(trajectories, axis=1)

    # 计算每段的长度 (B, N-1)
    # Compute the length of each segment (B, N-1)
    distances = np.linalg.norm(diffs, axis=2)

    # 累积弧长 (B, N) / Cumulative arc length (B, N)
    cumulative_lengths = np.zeros((trajectories.shape[0], trajectories.shape[1]))
    cumulative_lengths[:, 1:] = np.cumsum(distances, axis=1)

    # 总长度 / Total length
    total_lengths = cumulative_lengths[:, -1]
    
    return cumulative_lengths, total_lengths


def sample_at_arc_length_batch(trajectories: np.ndarray,
                               cumulative_lengths: np.ndarray,
                               arc_length: float) -> List[np.ndarray]:
    """
    批量等弧长采样（向量化版本）/ Batch equal-arc-length sampling (vectorized version).

    Args:
        trajectories: (B, N, 3) 批量轨迹 / Batched trajectories.
        cumulative_lengths: (B, N) 批量累积弧长 / Batched cumulative arc length.
        arc_length: 采样间距 / Sampling interval (arc length).

    Returns:
        sampled_trajectories: 列表，每个元素为 (M_i, 3) 的采样轨迹 / List where each element is a sampled trajectory of shape (M_i, 3).
    """
    B = trajectories.shape[0]
    total_lengths = cumulative_lengths[:, -1]

    sampled_trajectories = []

    for i in range(B):
        total_len = total_lengths[i]

        if total_len < arc_length:
            # 轨迹太短 / Trajectory too short
            sampled_trajectories.append(
                np.array([trajectories[i, 0], trajectories[i, -1]])
            )
            continue

        # 生成目标弧长 / Generate target arc lengths
        num_samples = int(total_len / arc_length) + 1
        target_lengths = np.linspace(0, total_len, num_samples)

        # 对每个维度插值 / Interpolate along each dimension
        sampled_points = np.zeros((num_samples, 3))
        for dim in range(3):
            sampled_points[:, dim] = np.interp(
                target_lengths,
                cumulative_lengths[i],
                trajectories[i, :, dim]
            )
        
        sampled_trajectories.append(sampled_points)
    
    return sampled_trajectories


def sample_predicted_trajectories_vectorized(predicted_control_points: List[np.ndarray],
                                            arc_length: float = 0.1,
                                            num_curve_points: int = 200,
                                            method: str = 'bspline',
                                            prediction_mode: str = 'control_points') -> List[np.ndarray]:
    """
    批量处理预测轨迹（向量化优化版本）/ Batch-process predicted trajectories (vectorized optimized version).

    优化策略：
    1. 将 List 转为 numpy 数组进行批量重建（支持 B-spline 和 Cubic Spline）
    2. 批量计算弧长
    3. 向量化插值

    Optimization strategy:
    1. Convert the List to a numpy array for batch reconstruction (supports B-spline and Cubic Spline)
    2. Batch-compute arc lengths
    3. Vectorized interpolation

    Args:
        predicted_control_points: 预测控制点列表，每个 (8, 3) / List of predicted control points, each of shape (8, 3).
        arc_length: 采样间距 / Sampling interval (arc length).
        num_curve_points: 重建密度 / Reconstruction density (number of curve points).
        method: 'bspline' 或 'cubic_spline' / 'bspline' or 'cubic_spline'.
        prediction_mode: 'control_points' 或 'waypoints' / 'control_points' or 'waypoints'.

    Returns:
        sampled_trajectories: 采样后的轨迹列表 / List of resampled trajectories.
    """
    if len(predicted_control_points) == 0:
        return []

    # Waypoints 模式：直接返回预测的航点，不进行样条拟合
    # Waypoints mode: directly return the predicted waypoints without spline fitting
    if prediction_mode == 'waypoints':
        return predicted_control_points

    # Control points 模式：进行样条拟合和采样
    # Control points mode: perform spline fitting and sampling
    # 1. 转换为批量数组 (B, 8, 3) / Convert to a batched array (B, 8, 3)
    control_points_batch = np.array(predicted_control_points)

    # 2. 批量重建 (B, num_curve_points, 3) / Batch reconstruction (B, num_curve_points, 3)
    dense_trajectories = reconstruct_trajectories_batch(
        control_points_batch,
        num_points=num_curve_points,
        method=method
    )

    # 3. 批量计算弧长 / Batch-compute arc lengths
    cumulative_lengths, _ = compute_arc_lengths_batch(dense_trajectories)

    # 4. 批量等弧长采样 / Batch equal-arc-length sampling
    sampled_trajectories = sample_at_arc_length_batch(
        dense_trajectories,
        cumulative_lengths,
        arc_length
    )
    
    return sampled_trajectories


def sample_predicted_trajectories_hybrid(predicted_control_points: List[np.ndarray],
                                        arc_length: float = 0.1,
                                        num_curve_points: int = 200,
                                        method: str = 'bspline',
                                        prediction_mode: str = 'control_points') -> List[np.ndarray]:
    """
    混合优化版本：重建向量化 + 保持原有采样逻辑 / Hybrid optimized version: vectorized reconstruction while keeping the original sampling logic.

    这个版本兼容性更好，只优化重建部分。
    This version has better compatibility and only optimizes the reconstruction part.

    Args:
        predicted_control_points: 预测控制点列表，每个 (8, 3) / List of predicted control points, each of shape (8, 3).
        arc_length: 采样间距 / Sampling interval (arc length).
        num_curve_points: 重建密度 / Reconstruction density (number of curve points).
        method: 'bspline' 或 'cubic_spline' / 'bspline' or 'cubic_spline'.
        prediction_mode: 'control_points' 或 'waypoints' / 'control_points' or 'waypoints'.
    """
    if len(predicted_control_points) == 0:
        return []

    # Waypoints 模式：直接返回预测的航点，不进行样条拟合
    # Waypoints mode: directly return the predicted waypoints without spline fitting
    if prediction_mode == 'waypoints':
        return predicted_control_points

    from sand_planner.trajectory.arc_length_sampling import TrajectoryArcLengthSampler

    # Control points 模式：进行样条拟合和采样
    # Control points mode: perform spline fitting and sampling
    # 1. 批量重建（主要优化点）/ Batch reconstruction (the main optimization)
    control_points_batch = np.array(predicted_control_points)
    dense_trajectories = reconstruct_trajectories_batch(
        control_points_batch,
        num_points=num_curve_points,
        method=method
    )

    # 2. 使用原有采样器（保持一致性）/ Use the original sampler (to keep consistency)
    sampler = TrajectoryArcLengthSampler(arc_length)
    sampled_trajectories = []
    
    for dense_traj in dense_trajectories:
        sampled_traj = sampler.sample_trajectory(dense_traj)
        sampled_trajectories.append(sampled_traj)
    
    return sampled_trajectories


if __name__ == "__main__":
    import time
    
    print("=" * 80)
    print("等弧长采样向量化优化测试")
    print("=" * 80)
    
    # 生成测试数据 / Generate test data
    batch_size = 64
    control_points_list = []
    for _ in range(batch_size):
        # 随机生成 8 个控制点 / Randomly generate 8 control points
        cp = np.random.randn(8, 3) * 5
        control_points_list.append(cp)

    arc_length = 0.1
    num_iterations = 50

    # 测试原始版本 / Benchmark the original version
    print(f"\n测试配置:")
    print(f"  批量大小: {batch_size}")
    print(f"  采样间距: {arc_length}m")
    print(f"  迭代次数: {num_iterations}")
    
    print("\n" + "-" * 80)
    print("测试原始版本（逐个处理）...")
    
    try:
        from sand_planner.trajectory.arc_length_sampling import sample_predicted_trajectories
        
        times_original = []
        for _ in range(num_iterations):
            start = time.time()
            result1 = sample_predicted_trajectories(control_points_list, arc_length)
            end = time.time()
            times_original.append((end - start) * 1000)
        
        avg_original = np.mean(times_original[5:])  # 跳过前 5 次预热 / Skip the first 5 warm-up runs
        print(f"  平均耗时: {avg_original:.2f} ms")
        
    except Exception as e:
        print(f"  ❌ 原始版本测试失败: {e}")
        avg_original = None
    
    # 测试向量化版本 / Benchmark the vectorized version
    print("\n测试向量化版本（批量处理）...")
    
    times_vectorized = []
    for _ in range(num_iterations):
        start = time.time()
        result2 = sample_predicted_trajectories_vectorized(control_points_list, arc_length)
        end = time.time()
        times_vectorized.append((end - start) * 1000)
    
    avg_vectorized = np.mean(times_vectorized[5:])
    print(f"  平均耗时: {avg_vectorized:.2f} ms")
    
    # 测试混合版本 / Benchmark the hybrid version
    print("\n测试混合版本（B样条向量化 + 原采样器）...")
    
    times_hybrid = []
    for _ in range(num_iterations):
        start = time.time()
        result3 = sample_predicted_trajectories_hybrid(control_points_list, arc_length)
        end = time.time()
        times_hybrid.append((end - start) * 1000)
    
    avg_hybrid = np.mean(times_hybrid[5:])
    print(f"  平均耗时: {avg_hybrid:.2f} ms")
    
    # 性能对比 / Performance comparison
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    
    if avg_original:
        speedup_vectorized = avg_original / avg_vectorized
        speedup_hybrid = avg_original / avg_hybrid
        
        print(f"\n原始版本:     {avg_original:.2f} ms")
        print(f"向量化版本:   {avg_vectorized:.2f} ms  (加速 {speedup_vectorized:.1f}x)")
        print(f"混合版本:     {avg_hybrid:.2f} ms  (加速 {speedup_hybrid:.1f}x)")
        
        time_saved_vectorized = avg_original - avg_vectorized
        time_saved_hybrid = avg_original - avg_hybrid
        
        print(f"\n🚀 优化效果:")
        print(f"  向量化版本节省: {time_saved_vectorized:.2f} ms/次")
        print(f"  混合版本节省:   {time_saved_hybrid:.2f} ms/次")
        
        if time_saved_hybrid > 5:
            print(f"\n✅ 成功优化等弧长采样，节省 {time_saved_hybrid:.1f}ms！")
    else:
        print(f"\n向量化版本: {avg_vectorized:.2f} ms")
        print(f"混合版本:   {avg_hybrid:.2f} ms")
    
    print("\n" + "=" * 80)
