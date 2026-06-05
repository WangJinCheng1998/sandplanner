#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹优化模块 - 三次样条插值 / Trajectory optimization module - cubic spline interpolation.

基于 iPlanner 的三次样条 (Cubic Spline) 实现，适配 SanD-planner，
用于从稀疏控制点生成平滑的密集轨迹。
Cubic spline implementation based on iPlanner, adapted for SanD-planner,
used to generate smooth dense trajectories from sparse control points.
"""

import torch
torch.set_default_dtype(torch.float32)


class CubicSplineTorch:
    """
    PyTorch 实现的三次样条插值 / Cubic spline interpolation implemented in PyTorch.

    参考 / Reference:
    https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    """
    def __init__(self):
        return None

    def h_poly(self, t):
        """Hermite 多项式基函数 / Hermite polynomial basis functions."""
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :] ** alpha[None, :, None]
        A = torch.tensor([
            [1, 0, -3, 2], 
            [0, 1, -2, 1], 
            [0, 0, 3, -2], 
            [0, 0, -1, 1]
        ], dtype=t.dtype, device=t.device)
        return A @ tt

    def interp(self, x, y, xs):
        """
        三次样条插值 / Cubic spline interpolation.

        Args:
            x: (batch_size, num_control_points) 控制点参数位置 / parametric positions of control points.
            y: (batch_size, num_control_points, dims) 控制点值 / control point values.
            xs: (batch_size, num_samples) 采样点参数位置 / parametric positions of sample points.

        Returns:
            out: (batch_size, num_samples, dims) 插值后的轨迹点 / interpolated trajectory points.
        """
        # 计算一阶导数（斜率） / Compute first-order derivatives (slopes).
        m = (y[:, 1:, :] - y[:, :-1, :]) / torch.unsqueeze(x[:, 1:] - x[:, :-1], 2)
        # 用中心差分近似中间点的导数，首尾使用单侧差分
        # Approximate interior derivatives with central differences; use one-sided differences at the ends.
        m = torch.cat([m[:, None, 0], (m[:, 1:] + m[:, :-1]) / 2, m[:, None, -1]], 1)

        # 找到每个采样点对应的区间 / Locate the interval that contains each sample point.
        idxs = torch.searchsorted(x[0, 1:], xs[0, :])
        dx = x[:, idxs + 1] - x[:, idxs]

        # 计算 Hermite 基函数 / Evaluate the Hermite basis functions.
        hh = self.h_poly((xs - x[:, idxs]) / dx)
        hh = torch.transpose(hh, 1, 2)

        # 组合四个 Hermite 基函数 / Combine the four Hermite basis functions.
        out = hh[:, :, 0:1] * y[:, idxs, :]
        out = out + hh[:, :, 1:2] * m[:, idxs] * dx[:, :, None]
        out = out + hh[:, :, 2:3] * y[:, idxs + 1, :]
        out = out + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:, :, None]
        return out


class TrajOpt:
    """
    轨迹优化类 - 使用三次样条 (Cubic Spline) 生成平滑轨迹 /
    Trajectory optimization class - generates smooth trajectories using cubic splines.
    """
    def __init__(self):
        self.cs_interp = CubicSplineTorch()
        return None

    def generate_trajectory_from_control_points(self, control_points, step=0.1):
        """
        从控制点生成平滑轨迹 / Generate a smooth trajectory from control points.

        Args:
            control_points: (batch_size, num_points, dims) 控制点，不包含起点 /
                control points, excluding the start point.
            step: float 采样步长，决定输出轨迹的密度 /
                sampling step that determines the density of the output trajectory.

        Returns:
            waypoints: (batch_size, num_samples, dims) 密集轨迹点 / dense trajectory points.
        """
        batch_size, num_p, dims = control_points.shape

        # 在起点处添加原点 (0, 0, 0) / Prepend the origin (0, 0, 0) as the start point.
        origin = torch.zeros(batch_size, 1, dims, device=control_points.device,
                            requires_grad=control_points.requires_grad)
        points_with_origin = torch.cat((origin, control_points), dim=1)
        num_p = num_p + 1

        # 创建参数空间的采样点 / Create sample points in the parameter space.
        # 确保 xs 不超过 num_p - 1 - epsilon，避免边界索引错误
        # Keep xs below num_p - 1 - epsilon to avoid out-of-bounds indexing.
        max_param = num_p - 1 - 1e-6  # 略小于最大参数值，避免边界问题 / slightly below the max to avoid boundary issues
        xs = torch.arange(0, max_param + step, step, device=control_points.device)
        # 裁剪确保不超过边界 / Clamp to stay within bounds.
        xs = torch.clamp(xs, 0, max_param)
        xs = xs.repeat(batch_size, 1)

        # 控制点的参数位置 [0, 1, 2, ..., num_p-1] / Parametric positions of control points.
        x = torch.arange(num_p, device=control_points.device, dtype=control_points.dtype)
        x = x.repeat(batch_size, 1)

        # 使用三次样条插值 / Run the cubic spline interpolation.
        waypoints = self.cs_interp.interp(x, points_with_origin, xs)

        return waypoints

    def TrajGeneratorFromPFreeRot(self, preds, step=0.1):
        """
        兼容 iPlanner 的接口 / iPlanner-compatible interface.

        Args:
            preds: (batch_size, num_points, dims) 预测的控制点 / predicted control points.
            step: float 采样步长 / sampling step.

        Returns:
            waypoints: (batch_size, num_samples, dims) 生成的轨迹 / generated trajectory.
        """
        return self.generate_trajectory_from_control_points(preds, step)


def sample_cubic_spline_trajectory(
    control_points: torch.Tensor,
    num_samples: int = 50,
    dimensions: int = 3
) -> torch.Tensor:
    """
    便捷函数：从控制点采样固定数量的轨迹点 /
    Convenience function: sample a fixed number of trajectory points from control points.

    Args:
        control_points: (batch_size, num_control_points, dims) 控制点 / control points.
        num_samples: int 目标采样点数量 / target number of sample points.
        dimensions: int 轨迹维度（2D 或 3D） / trajectory dimensionality (2D or 3D).

    Returns:
        trajectory: (batch_size, num_samples, dims) 采样后的轨迹 / resampled trajectory.
    """
    traj_opt = TrajOpt()
    batch_size, num_cp, dims = control_points.shape

    # 计算采样步长 / Compute the sampling step.
    step = (num_cp) / num_samples

    # 生成轨迹 / Generate the trajectory.
    trajectory = traj_opt.generate_trajectory_from_control_points(control_points, step)

    # 确保输出正好是 num_samples 个点 / Ensure the output has exactly num_samples points.
    if trajectory.shape[1] != num_samples:
        # 使用线性插值调整到目标数量 / Resample to the target count via linear interpolation.
        trajectory = torch.nn.functional.interpolate(
            trajectory.permute(0, 2, 1),  # (B, dims, N)
            size=num_samples,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)  # (B, N, dims)

    return trajectory


if __name__ == "__main__":
    """测试代码"""
    print("测试 Cubic Spline 轨迹生成...")
    
    # 创建测试数据 / Create test data.
    batch_size = 2
    num_control_points = 8
    dims = 3

    # 生成随机控制点（不包含起点） / Generate random control points (excluding the start point).
    control_points = torch.randn(batch_size, num_control_points, dims) * 2.0
    control_points[:, :, 0] = torch.cumsum(torch.abs(control_points[:, :, 0]), dim=1)  # 确保 x 递增 / ensure x is increasing
    
    print(f"控制点形状: {control_points.shape}")
    print(f"控制点范围: [{control_points.min():.2f}, {control_points.max():.2f}]")
    
    # 测试 1: 使用 step 参数 / Test 1: use the step argument.
    traj_opt = TrajOpt()
    trajectory1 = traj_opt.generate_trajectory_from_control_points(control_points, step=0.2)
    print(f"\n使用step=0.2:")
    print(f"  输出轨迹形状: {trajectory1.shape}")
    print(f"  轨迹点数: {trajectory1.shape[1]}")
    
    # 测试 2: 使用固定采样数 / Test 2: use a fixed number of samples.
    num_samples = 50
    trajectory2 = sample_cubic_spline_trajectory(control_points, num_samples=num_samples)
    print(f"\n使用固定采样数={num_samples}:")
    print(f"  输出轨迹形状: {trajectory2.shape}")
    print(f"  轨迹点数: {trajectory2.shape[1]}")
    
    # 测试 3: GPU 支持 / Test 3: GPU support.
    if torch.cuda.is_available():
        control_points_gpu = control_points.cuda()
        trajectory_gpu = sample_cubic_spline_trajectory(control_points_gpu, num_samples=50)
        print(f"\nGPU测试:")
        print(f"  输出轨迹形状: {trajectory_gpu.shape}")
        print(f"  设备: {trajectory_gpu.device}")
    
    print("\n✅ Cubic Spline 测试完成!")
