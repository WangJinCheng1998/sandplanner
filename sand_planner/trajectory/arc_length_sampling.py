#!/usr/bin/env python3
"""
轨迹等弧长采样模块 / Trajectory arc-length sampling module.

对 B-spline 拟合的轨迹进行等弧长采样，得到离散控制点。
Performs arc-length-uniform sampling on a B-spline-fitted trajectory to obtain
discrete control points.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional
import warnings

class TrajectoryArcLengthSampler:
    """轨迹等弧长采样器 / Arc-length-uniform trajectory sampler."""

    def __init__(self, arc_length: float = 0.1):
        """
        初始化采样器 / Initialize the sampler.

        Args:
            arc_length: 采样间距（米） / Sampling spacing in meters.
        """
        self.arc_length = arc_length
    
    def sample_trajectory(self, trajectory: np.ndarray, 
                         arc_length: Optional[float] = None) -> np.ndarray:
        """
        对轨迹进行等弧长采样 / Resample a trajectory at uniform arc-length intervals.

        Args:
            trajectory: 轨迹点 [N, 3] (x, y, z) / Trajectory points [N, 3] (x, y, z).
            arc_length: 采样间距；若为 None 则使用初始化值 / Sampling spacing; if None, use the value set at init.

        Returns:
            sampled_points: 等弧长采样点 [M, 3] / Arc-length-uniform sampled points [M, 3].
        """
        if arc_length is None:
            arc_length = self.arc_length
            
        if len(trajectory) < 2:
            return trajectory.copy()
        
        # 1. 计算累积弧长 / Compute cumulative arc length.
        cumulative_lengths = self._compute_cumulative_arc_length(trajectory)
        total_length = cumulative_lengths[-1]

        if total_length < arc_length:
            # 轨迹太短，返回起点和终点 / Trajectory too short; return only the start and end points.
            return np.array([trajectory[0], trajectory[-1]])

        # 2. 生成等间距的弧长采样点 / Generate evenly spaced arc-length sample positions.
        num_samples = int(total_length / arc_length) + 1
        target_lengths = np.linspace(0, total_length, num_samples)

        # 3. 插值得到对应的 3D 点 / Interpolate the corresponding 3D points.
        sampled_points = self._interpolate_points_at_lengths(
            trajectory, cumulative_lengths, target_lengths
        )
        
        return sampled_points
    
    def sample_multiple_trajectories(self, trajectories: List[np.ndarray],
                                   arc_length: Optional[float] = None) -> List[np.ndarray]:
        """
        批量处理多条轨迹 / Resample multiple trajectories in a batch.

        Args:
            trajectories: 轨迹列表，每个元素为 [N_i, 3] / List of trajectories, each shaped [N_i, 3].
            arc_length: 采样间距 / Sampling spacing.

        Returns:
            sampled_trajectories: 采样后的轨迹列表 / List of resampled trajectories.
        """
        return [self.sample_trajectory(traj, arc_length) for traj in trajectories]
    
    def _compute_cumulative_arc_length(self, trajectory: np.ndarray) -> np.ndarray:
        """计算累积弧长 / Compute the cumulative arc length along a trajectory."""
        if len(trajectory) < 2:
            return np.array([0.0])

        # 计算相邻点之间的距离 / Compute distances between consecutive points.
        diffs = np.diff(trajectory, axis=0)  # [N-1, 3]
        distances = np.linalg.norm(diffs, axis=1)  # [N-1]

        # 累积弧长 / Cumulative arc length.
        cumulative_lengths = np.zeros(len(trajectory))
        cumulative_lengths[1:] = np.cumsum(distances)
        
        return cumulative_lengths
    
    def _interpolate_points_at_lengths(self, trajectory: np.ndarray,
                                     cumulative_lengths: np.ndarray,
                                     target_lengths: np.ndarray) -> np.ndarray:
        """在指定弧长位置插值得到 3D 点 / Interpolate 3D points at the given arc-length positions."""

        # 确保目标长度在有效范围内 / Clamp target lengths to the valid range.
        target_lengths = np.clip(target_lengths, 0, cumulative_lengths[-1])

        # 对每个维度分别插值 / Interpolate each coordinate axis separately.
        sampled_points = np.zeros((len(target_lengths), 3))

        for dim in range(3):  # x, y, z
            # 创建插值函数 / Build the interpolation function.
            interp_func = interp1d(
                cumulative_lengths, 
                trajectory[:, dim],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # 插值 / Evaluate the interpolant.
            sampled_points[:, dim] = interp_func(target_lengths)
        
        return sampled_points


def sample_bspline_trajectory(control_points: np.ndarray, 
                            arc_length: float = 0.1,
                            num_curve_points: int = 200) -> np.ndarray:
    """
    便捷函数：从 B-spline 控制点生成等弧长采样轨迹。
    Convenience helper that generates an arc-length-uniform sampled trajectory
    from B-spline control points.

    Args:
        control_points: B-spline 控制点 [8, 3] / B-spline control points [8, 3].
        arc_length: 采样间距（米） / Sampling spacing in meters.
        num_curve_points: 生成曲线的密度 / Density of the reconstructed dense curve.

    Returns:
        sampled_trajectory: 等弧长采样点 [M, 3] / Arc-length-uniform sampled points [M, 3].
    """
    # 导入 B-spline 重建函数 / Import the B-spline reconstruction function.
    try:
        from sand_planner.utils.bspline import reconstruct_trajectory_from_8cp

        # 重建高密度轨迹 / Reconstruct a high-density trajectory.
        dense_trajectory = reconstruct_trajectory_from_8cp(
            control_points, num_points=num_curve_points
        )

        # 等弧长采样 / Arc-length-uniform sampling.
        sampler = TrajectoryArcLengthSampler(arc_length)
        sampled_trajectory = sampler.sample_trajectory(dense_trajectory)

        return sampled_trajectory

    except ImportError:
        print("警告：未找到bspline_core模块，使用线性插值")
        # 降级到线性插值 / Fall back to linear interpolation.
        sampler = TrajectoryArcLengthSampler(arc_length)
        return sampler.sample_trajectory(control_points)


def sample_predicted_trajectories(predicted_control_points: List[np.ndarray],
                                arc_length: float = 0.1) -> List[np.ndarray]:
    """
    批量处理预测的轨迹控制点 / Resample a batch of predicted trajectory control points.

    Args:
        predicted_control_points: 预测的控制点列表，每个元素为 [8, 3] / List of predicted control points, each shaped [8, 3].
        arc_length: 采样间距（米） / Sampling spacing in meters.

    Returns:
        sampled_trajectories: 等弧长采样轨迹列表 / List of arc-length-uniform sampled trajectories.
    """
    sampled_trajectories = []

    for i, cp in enumerate(predicted_control_points):
        try:
            sampled_traj = sample_bspline_trajectory(cp, arc_length)
            sampled_trajectories.append(sampled_traj)
            # 详细的逐条输出（已禁用） / Per-trajectory verbose output (disabled).
            # print(f"轨迹 {i+1}: 采样得到 {len(sampled_traj)} 个点")
        except Exception as e:
            print(f"警告：轨迹 {i+1} 采样失败: {e}")
            # 使用控制点作为备选 / Fall back to the raw control points.
            sampler = TrajectoryArcLengthSampler(arc_length)
            sampled_traj = sampler.sample_trajectory(cp)
            sampled_trajectories.append(sampled_traj)
    
    return sampled_trajectories


def analyze_sampling_quality(original_trajectory: np.ndarray,
                           sampled_trajectory: np.ndarray) -> dict:
    """
    分析采样质量 / Analyze the quality of an arc-length sampling.

    Args:
        original_trajectory: 原始密集轨迹 [N, 3] / Original dense trajectory [N, 3].
        sampled_trajectory: 采样后轨迹 [M, 3] / Resampled trajectory [M, 3].

    Returns:
        analysis: 分析结果字典 / Dictionary of analysis metrics.
    """
    # 计算原始轨迹长度 / Compute the original and sampled trajectory lengths.
    orig_lengths = TrajectoryArcLengthSampler()._compute_cumulative_arc_length(original_trajectory)
    sampled_lengths = TrajectoryArcLengthSampler()._compute_cumulative_arc_length(sampled_trajectory)

    orig_total_length = orig_lengths[-1]
    sampled_total_length = sampled_lengths[-1]

    # 计算采样间距的均匀性 / Evaluate the uniformity of the sample spacing.
    if len(sampled_trajectory) > 1:
        sampled_diffs = np.diff(sampled_trajectory, axis=0)
        sampled_distances = np.linalg.norm(sampled_diffs, axis=1)
        distance_std = np.std(sampled_distances)
        distance_mean = np.mean(sampled_distances)
    else:
        distance_std = 0
        distance_mean = 0
    
    analysis = {
        'original_points': len(original_trajectory),
        'sampled_points': len(sampled_trajectory),
        'compression_ratio': len(original_trajectory) / len(sampled_trajectory) if len(sampled_trajectory) > 0 else 0,
        'original_length': orig_total_length,
        'sampled_length': sampled_total_length,
        'length_error': abs(orig_total_length - sampled_total_length),
        'avg_sample_distance': distance_mean,
        'sample_distance_std': distance_std,
        'uniformity_score': 1.0 - (distance_std / distance_mean) if distance_mean > 0 else 0
    }
    
    return analysis


def demo_arc_length_sampling():
    """演示等弧长采样功能 / Demonstrate the arc-length sampling functionality."""

    print("=== 轨迹等弧长采样演示 ===\n")

    # 创建示例 B-spline 控制点 / Create example B-spline control points.
    demo_control_points = np.array([
        [0.0, 0.0, 0.0],   # 起点 / Start point.
        [0.5, 0.1, 0.0],
        [1.0, 0.3, 0.1],
        [1.5, 0.2, 0.2],
        [2.0, -0.1, 0.3],
        [2.5, -0.3, 0.2],
        [3.0, -0.2, 0.1],
        [3.5, 0.0, 0.0],   # 终点 / End point.
    ])

    print("1. 生成示例轨迹...")
    print(f"控制点形状: {demo_control_points.shape}")

    # 进行等弧长采样 / Perform arc-length-uniform sampling.
    sampled_trajectory = sample_bspline_trajectory(
        demo_control_points, 
        arc_length=0.1
    )
    
    # 可选：打印采样结果概览 / Optional: print an overview of the sampled result.
    # print(f"采样结果: {len(sampled_trajectory)} 个点")
    # print(f"前5个采样点:")
    # for i, point in enumerate(sampled_trajectory[:5]):
    #     print(f"  点{i+1}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

    # 可选：分析采样质量 / Optional: analyze the sampling quality.
    # try:
    #     from sand_planner.utils.bspline import reconstruct_trajectory_from_8cp
    #     dense_trajectory = reconstruct_trajectory_from_8cp(demo_control_points, 200)

    #     analysis = analyze_sampling_quality(dense_trajectory, sampled_trajectory)

        # print("\n2. 采样质量分析:")
        # print(f"  原始点数: {analysis['original_points']}")
        # print(f"  采样点数: {analysis['sampled_points']}")
        # print(f"  压缩比: {analysis['compression_ratio']:.1f}:1")
        # print(f"  轨迹长度: {analysis['original_length']:.3f}m")
        # print(f"  平均采样间距: {analysis['avg_sample_distance']:.3f}m")
        # print(f"  间距标准差: {analysis['sample_distance_std']:.4f}m")
        # print(f"  均匀性得分: {analysis['uniformity_score']:.3f} (1.0为完全均匀)")

    # except ImportError:
    #     print("未找到bspline_core模块，跳过质量分析")

    # 可选：对比不同采样间距 / Optional: compare different sampling spacings.
    # print("\n3. 不同采样间距对比:")
    # test_intervals = [0.05, 0.1, 0.2, 0.5]

    # for interval in test_intervals:
    #     sampled = sample_bspline_trajectory(demo_control_points, interval)
    #     print(f"  间距{interval}m: {len(sampled)}个点")

    return sampled_trajectory


if __name__ == "__main__":
    # 运行演示 / Run the demonstration.
    demo_arc_length_sampling()
    
    print("\n=== 使用示例 ===")
    print("""
# 基本使用
from sand_planner.trajectory.arc_length_sampling import sample_bspline_trajectory

# 对单条轨迹采样
control_points = your_predicted_control_points  # [8, 3]
sampled_points = sample_bspline_trajectory(control_points, arc_length=0.1)

# 批量处理多条轨迹
from sand_planner.trajectory.arc_length_sampling import sample_predicted_trajectories

all_control_points = [cp1, cp2, cp3, ...]  # 预测的多条轨迹
sampled_trajectories = sample_predicted_trajectories(all_control_points, arc_length=0.1)

# 自定义采样器
from sand_planner.trajectory.arc_length_sampling import TrajectoryArcLengthSampler

sampler = TrajectoryArcLengthSampler(arc_length=0.05)
result = sampler.sample_trajectory(your_dense_trajectory)
    """)
