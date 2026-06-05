#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹绘制函数 —— 在深度图像上绘制生成的轨迹。
基于 draw.py 的投影方式，适配真实的相机参数和坐标系。

Trajectory drawing utilities for overlaying generated trajectories on depth images.
Uses the projection scheme from draw.py, adapted to the real camera intrinsics and coordinate frames.
"""

import numpy as np
import cv2

# 设置 matplotlib 使用非交互式后端，避免 GUI 相关的错误
# Configure matplotlib to use a non-interactive backend to avoid GUI-related errors
import matplotlib
# 使用 Anti-Grain Geometry 后端，不需要 X11 或其他 GUI
# Use the Anti-Grain Geometry backend, which requires no X11 or other GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import torch

def project_trajectory_to_image(trajectory_points, camera_intrinsics, camera_extrinsics=None):
    """
    将 3D 轨迹点投影到图像平面 / Project 3D trajectory points onto the image plane.

    参数:
        trajectory_points: (N, 3) baselink 坐标系下的 3D 轨迹点 [x, y, z] /
            (N, 3) 3D trajectory points [x, y, z] in the baselink frame.
        camera_intrinsics: 相机内参字典 {'fx', 'fy', 'ppx', 'ppy'} /
            Camera intrinsics dict {'fx', 'fy', 'ppx', 'ppy'}.
        camera_extrinsics: 相机外参；为 None 时使用默认基准坐标系变换 /
            Camera extrinsics; if None, the default base-frame transform is used.

    返回:
        uvz: (M, 3) 有效投影点 [u, v, depth_z]，仅包含相机前方的点 /
            (M, 3) valid projected points [u, v, depth_z], only those in front of the camera.
        mask: (N,) 布尔掩码，指示哪些点被成功投影 /
            (N,) boolean mask indicating which points were successfully projected.
    """
    # 默认外参：相机位于 baselink 前方 0.25m、上方 0.10m
    # Default extrinsics: camera is 0.25m in front of and 0.10m above the baselink
    # URDF: <origin xyz="0.25 0 0.10" rpy="0 0 0"/>
    # 注意：此处假设轨迹点相对于 baselink；若为地面坐标，还需额外考虑 baselink 高度
    # Note: trajectory points are assumed relative to the baselink; if they are in ground
    # coordinates, the baselink height must additionally be accounted for
    # 使用正确的相机偏移 / Use the correct camera offset
    t_bc = np.array([0.25, 0.0, 0.10], dtype=np.float64)
    # baselink（x 前、y 左、z 上）到相机光学坐标系（x 右、y 下、z 前）的旋转
    # Rotation from baselink (x forward, y left, z up) to the camera optical frame
    # (x right, y down, z forward)
    R_ol = np.array([[0, -1,  0],
                     [0,  0, -1], 
                     [1,  0,  0]], dtype=np.float64)
    
    # 步骤 1：baselink -> camera_link（仅平移，因为坐标系平行）
    # Step 1: baselink -> camera_link (translation only, since the frames are parallel)
    P_camera = trajectory_points - t_bc  # (N, 3)

    # 步骤 2：camera_link -> camera_optical（旋转）
    # Step 2: camera_link -> camera_optical (rotation)
    P_optical = (R_ol @ P_camera.T).T  # (N, 3)

    # 步骤 3：过滤相机前方的点 (Z > 0)
    # Step 3: keep only the points in front of the camera (Z > 0)
    Z = P_optical[:, 2]
    mask = Z > 0
    P_optical_valid = P_optical[mask]
    Z_valid = Z[mask]
    
    if len(P_optical_valid) == 0:
        return np.array([]).reshape(0, 3), mask
    
    # 步骤 4：投影到图像平面
    # Step 4: project onto the image plane
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    ppx = camera_intrinsics['ppx']
    ppy = camera_intrinsics['ppy']
    
    u = ppx + fx * (P_optical_valid[:, 0] / Z_valid)
    v = ppy + fy * (P_optical_valid[:, 1] / Z_valid)
    
    uvz = np.stack([u, v, Z_valid], axis=1)
    
    return uvz, mask

def draw_trajectory_on_depth_image(depth_image, trajectory_points, camera_intrinsics, 
                                  color=(255, 255, 255), thickness=2, point_size=3):
    """
    在深度图像上绘制单条轨迹 / Draw a single trajectory on a depth image.

    参数:
        depth_image: (H, W) 归一化深度图像 [0, 1] / (H, W) normalized depth image in [0, 1].
        trajectory_points: (N, 3) 3D 轨迹点 / (N, 3) 3D trajectory points.
        camera_intrinsics: 相机内参 / Camera intrinsics.
        color: 轨迹颜色 (R, G, B) / Trajectory color (R, G, B).
        thickness: 线条粗细 / Line thickness.
        point_size: 控制点大小 / Control point size.

    返回:
        result_image: (H, W, 3) RGB 图像，轨迹叠加在深度图上 /
            (H, W, 3) RGB image with the trajectory overlaid on the depth image.
    """
    # 将深度图转换为可视化图像 / Convert the depth map into a visualization image
    depth_vis = (depth_image * 255).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)

    # 投影轨迹点 / Project the trajectory points
    uvz, mask = project_trajectory_to_image(trajectory_points, camera_intrinsics)

    if len(uvz) < 2:
        print("警告: 没有足够的投影点来绘制轨迹")
        return depth_rgb

    # 获取图像尺寸 / Get the image dimensions
    H, W = depth_image.shape

    # 绘制轨迹线 / Draw the trajectory line
    uv_points = uvz[:, :2].astype(int)
    depths = uvz[:, 2]

    # 过滤图像边界内的点 / Keep only the points inside the image bounds
    valid_mask = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) &
                  (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))

    if valid_mask.sum() < 2:
        print("警告: 没有足够的有效点在图像范围内")
        return depth_rgb

    valid_points = uv_points[valid_mask]
    valid_depths = depths[valid_mask]

    # 绘制连续线段，线条粗细随深度调整（近粗远细）
    # Draw connected segments with thickness scaled by depth (thicker when near, thinner when far)
    for i in range(len(valid_points) - 1):
        pt1 = tuple(valid_points[i])
        pt2 = tuple(valid_points[i + 1])

        # 根据深度调整线条粗细（近处粗，远处细）
        # Adjust line thickness by depth (thicker near, thinner far)
        depth_avg = (valid_depths[i] + valid_depths[i + 1]) / 2
        dynamic_thickness = max(1, int(thickness * (3.0 / (depth_avg + 0.5))))

        cv2.line(depth_rgb, pt1, pt2, color, dynamic_thickness)

    # 绘制控制点（当输入本身就是控制点时）
    # Draw control points (when the input is itself a set of control points)
    # 假设少于 10 个点即为控制点 / Assume fewer than 10 points indicates control points
    if len(trajectory_points) <= 10:
        for i, (u, v, depth) in enumerate(uvz):
            u, v = int(u), int(v)
            if 0 <= u < W and 0 <= v < H:
                # 根据深度调整点的大小 / Adjust point size by depth
                dynamic_size = max(2, int(point_size * (3.0 / (depth + 0.5))))
                cv2.circle(depth_rgb, (u, v), dynamic_size, color, -1)

                # 可选：添加点的编号 / Optional: annotate each point with its index
                cv2.putText(depth_rgb, str(i), (u + 5, v - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return depth_rgb

def visualize_multiple_trajectories_on_depth(depth_image, trajectory_list, camera_intrinsics,
                                           trajectory_labels=None, save_path=None, 
                                           title="Trajectories on Depth Image"):
    """
    在深度图像上可视化多条轨迹 / Visualize multiple trajectories on a depth image.

    参数:
        depth_image: (H, W) 归一化深度图像 / (H, W) normalized depth image.
        trajectory_list: 轨迹列表，每个元素是 (N, 3) 的轨迹点 /
            List of trajectories, each element being (N, 3) trajectory points.
        camera_intrinsics: 相机内参 / Camera intrinsics.
        trajectory_labels: 轨迹标签列表 / List of trajectory labels.
        save_path: 保存路径 / Output save path.
        title: 图像标题 / Figure title.

    返回:
        fig: matplotlib 图像对象 / The matplotlib figure object.
    """
    # 预定义颜色列表 / Predefined color list
    colors = [
        (255, 0, 0),    # 红色 / red
        (0, 255, 0),    # 绿色 / green
        (0, 0, 255),    # 蓝色 / blue
        (255, 255, 0),  # 黄色 / yellow
        (255, 0, 255),  # 紫色 / magenta
        (0, 255, 255),  # 青色 / cyan
        (255, 165, 0),  # 橙色 / orange
        (128, 0, 128),  # 紫罗兰 / violet
        (255, 192, 203), # 粉色 / pink
        (128, 128, 128), # 灰色 / gray
        (165, 42, 42),   # 棕色 / brown
        (0, 128, 0),     # 深绿色 / dark green
    ]

    # 创建基础深度图像 / Create the base depth image
    depth_vis = (depth_image * 255).astype(np.uint8)
    result_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)

    # 绘制每条轨迹 / Draw each trajectory
    trajectory_info = []

    for i, trajectory in enumerate(trajectory_list):
        color = colors[i % len(colors)]
        label = trajectory_labels[i] if trajectory_labels else f"Trajectory {i+1}"

        # 投影并绘制轨迹 / Project and draw the trajectory
        uvz, mask = project_trajectory_to_image(trajectory, camera_intrinsics)

        if len(uvz) < 2:
            print(f"警告: 轨迹 {label} 没有足够的投影点")
            continue

        # 绘制轨迹线 / Draw the trajectory line
        H, W = depth_image.shape
        uv_points = uvz[:, :2].astype(int)
        depths = uvz[:, 2]

        # 过滤有效点 / Keep only the valid points
        valid_mask = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) &
                      (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))

        if valid_mask.sum() < 2:
            continue

        valid_points = uv_points[valid_mask]
        valid_depths = depths[valid_mask]

        # 绘制连续线段 / Draw connected segments
        thickness = 2
        for j in range(len(valid_points) - 1):
            pt1 = tuple(valid_points[j])
            pt2 = tuple(valid_points[j + 1])

            # 根据深度调整线条粗细 / Adjust line thickness by depth
            depth_avg = (valid_depths[j] + valid_depths[j + 1]) / 2
            dynamic_thickness = max(1, int(thickness * (3.0 / (depth_avg + 0.5))))

            cv2.line(result_image, pt1, pt2, color, dynamic_thickness)

        # 记录轨迹信息 / Record trajectory info
        traj_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        max_y_offset = np.abs(trajectory[:, 1]).max()
        
        trajectory_info.append({
            'label': label,
            'color': color,
            'length': traj_length,
            'max_y_offset': max_y_offset,
            'num_projected_points': len(valid_points)
        })
    
    # 使用 matplotlib 显示结果 / Display the results with matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左侧：原始深度图 / Left: the original depth image
    ax1 = axes[0]
    im1 = ax1.imshow(depth_image, cmap='viridis')
    ax1.set_title('Original Depth Image')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Normalized Depth')

    # 右侧：带轨迹的图像 / Right: the image with trajectories
    ax2 = axes[1]
    # OpenCV 使用 BGR，matplotlib 使用 RGB，需要转换
    # OpenCV uses BGR while matplotlib uses RGB, so a conversion is needed
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(result_rgb)
    ax2.set_title(title)
    ax2.axis('off')

    # 添加轨迹信息文本 / Add the trajectory info text
    info_text = "Trajectory Info:\n"
    for info in trajectory_info:
        color_rgb = (info['color'][2], info['color'][1], info['color'][0])  # BGR 转 RGB / BGR to RGB
        info_text += f"• {info['label']}: L={info['length']:.2f}m, "
        info_text += f"Y_max={info['max_y_offset']:.3f}m\n"
    
    ax2.text(1.02, 0.98, info_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果保存到: {save_path}")
    
    plt.show()
    
    return fig

def create_trajectory_from_control_points(control_points, num_points=100, method='bspline'):
    """
    从控制点重建轨迹 / Reconstruct a trajectory from control points.

    参数:
        control_points: (8, 3) 控制点 / (8, 3) control points.
        num_points: 重建轨迹的点数 / Number of points in the reconstructed trajectory.
        method: 'bspline' 或 'cubic_spline' / 'bspline' or 'cubic_spline'.

    返回:
        trajectory: (num_points, 3) 重建的轨迹点 / (num_points, 3) reconstructed trajectory points.
    """
    if method == 'cubic_spline':
        # 使用 Cubic Spline 重建 / Reconstruct using a cubic spline
        try:
            from sand_planner.utils.traj_opt import TrajOpt
            traj_opt = TrajOpt()

            # 转换为 torch 格式 / Convert to torch format
            # 注意：control_points 含 8 个点（包含原点），但 TrajOpt 会自动添加原点，
            # 因此需要去掉第一个点（原点）
            # Note: control_points has 8 points (including the origin), but TrajOpt adds the
            # origin automatically, so the first point (the origin) must be removed
            cp_without_origin = control_points[1:]  # (7, 3)
            cp_torch = torch.from_numpy(cp_without_origin).float().unsqueeze(0)  # (1, 7, 3)

            # 计算采样步长 / Compute the sampling step
            # TrajOpt 会添加原点，因此 7 个控制点 + 原点 = 8 个点，参数空间为 [0, 7]
            # TrajOpt adds the origin, so 7 control points + origin = 8 points, parameter space [0, 7]
            step = 7.0 / num_points

            with torch.no_grad():
                trajectory_torch = traj_opt.generate_trajectory_from_control_points(cp_torch, step=step)

            trajectory = trajectory_torch.squeeze(0).numpy()  # (M, 3)

            # 如果点数不匹配，使用插值对齐 / If the point count mismatches, align via interpolation
            if len(trajectory) != num_points:
                t_src = np.linspace(0, 1, len(trajectory))
                t_dst = np.linspace(0, 1, num_points)
                aligned = np.zeros((num_points, 3))
                for axis in range(3):
                    aligned[:, axis] = np.interp(t_dst, t_src, trajectory[:, axis])
                trajectory = aligned

            return trajectory

        except Exception as e:
            print(f"警告: Cubic Spline 重建失败: {e}, 回退到线性插值")
            # 回退到线性插值 / Fall back to linear interpolation
            t = np.linspace(0, 1, num_points)
            trajectory = np.zeros((num_points, 3))
            for i in range(3):
                trajectory[:, i] = np.interp(t, np.linspace(0, 1, len(control_points)), control_points[:, i])
            return trajectory
    else:
        # 使用 B-spline 重建 / Reconstruct using a B-spline
        try:
            from sand_planner.utils.bspline import reconstruct_trajectory_from_8cp

            return reconstruct_trajectory_from_8cp(control_points, num_points=num_points)
        except ImportError as e:
            print(f"警告: 无法导入B-spline重建函数: {e}")
            # 简单的线性插值作为后备方案 / Simple linear interpolation as a fallback
            t = np.linspace(0, 1, num_points)
            trajectory = np.zeros((num_points, 3))

            # 使用贝塞尔曲线近似 / Approximate with a Bezier-style interpolation
            for i in range(3):  # x, y, z 坐标 / x, y, z coordinates
                trajectory[:, i] = np.interp(t, np.linspace(0, 1, len(control_points)),
                                       control_points[:, i])

        return trajectory

# 测试函数 / Test function
def test_trajectory_drawing():
    """测试轨迹绘制功能 / Test the trajectory drawing functionality."""
    print("=== 测试轨迹绘制功能 ===")

    # 创建模拟深度图像 / Create a synthetic depth image
    H, W = 240, 320
    depth_image = np.random.rand(H, W) * 0.8 + 0.1  # 范围 [0.1, 0.9] / range [0.1, 0.9]

    # 下采样后的相机参数 (320×240) / Camera intrinsics after downsampling (320×240)
    camera_intrinsics = {
        'fx': 194.776,  # 389.551 * 0.5
        'fy': 194.776,  # 389.551 * 0.5
        'ppx': 162.106, # 324.211 * 0.5
        'ppy': 117.828  # 235.656 * 0.5
    }

    # 创建测试轨迹（简单的弯曲轨迹）/ Create test trajectories (simple curved paths)
    test_trajectories = []

    # 轨迹 1：直行 / Trajectory 1: straight ahead
    x = np.linspace(0, 3, 50)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    trajectory1 = np.stack([x, y, z], axis=1)
    test_trajectories.append(trajectory1)

    # 轨迹 2：向左弯曲 / Trajectory 2: curving left
    y = 0.3 * np.sin(x * np.pi / 3)
    trajectory2 = np.stack([x, y, z], axis=1)
    test_trajectories.append(trajectory2)

    # 轨迹 3：向右弯曲 / Trajectory 3: curving right
    y = -0.3 * np.sin(x * np.pi / 3)
    trajectory3 = np.stack([x, y, z], axis=1)
    test_trajectories.append(trajectory3)

    labels = ["Straight", "Left Curve", "Right Curve"]

    # 可视化 / Visualize
    visualize_multiple_trajectories_on_depth(
        depth_image, test_trajectories, camera_intrinsics,
        trajectory_labels=labels, save_path="test_trajectory_drawing.png",
        title="Test Trajectory Drawing"
    )

class Visualizer:
    """可视化器 / Trajectory visualizer."""

    def __init__(self, config):
        from sand_planner.config import InferenceConfig
        self.config = config

    def create_trajectory_from_control_points(self, control_points: np.ndarray, num_points: int = 100) -> np.ndarray:
        """从控制点创建轨迹（使用配置中指定的插值方法）/ Build a trajectory from control points using the interpolation method specified in the config."""
        # 使用配置中的 trajectory_interpolation 参数
        # Use the trajectory_interpolation parameter from the config
        return create_trajectory_from_control_points(control_points, num_points, method=self.config.trajectory_interpolation)

    def project_trajectory_to_image(self, trajectory: np.ndarray):
        """将轨迹投影到图像坐标 / Project a trajectory into image coordinates."""
        return project_trajectory_to_image(trajectory, self.config.camera_intrinsics)

    def visualize_trajectories(self, depth: np.ndarray, control_points_list,
                             best_index: int = 0, save_path=None):
        """可视化轨迹 / Visualize trajectories."""
        fig, ax = plt.subplots(figsize=(14, 10), frameon=False)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')

        # 显示深度图 / Show the depth image
        im = ax.imshow(depth, cmap='plasma', alpha=0.9)

        # 绘制轨迹 / Draw the trajectories
        for i, cp in enumerate(control_points_list):
            trajectory = self.create_trajectory_from_control_points(cp)
            uvz, mask = self.project_trajectory_to_image(trajectory)

            if len(uvz) > 1:
                u_coords = uvz[:, 0]
                v_coords = uvz[:, 1]

                if i == best_index:
                    # 最佳轨迹：绿色 / Best trajectory: green
                    ax.plot(u_coords, v_coords, color='lime', linewidth=4, alpha=1.0, zorder=10)
                    ax.plot(u_coords[0], v_coords[0], 'o', color='lime', markersize=8,
                           markeredgecolor='white', markeredgewidth=2, zorder=12)
                    ax.plot(u_coords[-1], v_coords[-1], 's', color='lime', markersize=8,
                           markeredgecolor='white', markeredgewidth=2, zorder=12)
                else:
                    # 其他轨迹：白色半透明 / Other trajectories: semi-transparent white
                    ax.plot(u_coords, v_coords, color='white', linewidth=2, alpha=0.4, zorder=5)
                    ax.plot(u_coords[0], v_coords[0], 'o', color='white', markersize=4, alpha=0.4, zorder=6)
                    ax.plot(u_coords[-1], v_coords[-1], 's', color='white', markersize=4, alpha=0.4, zorder=6)

        ax.set_xlim(0, depth.shape[1])
        ax.set_ylim(depth.shape[0], 0)

        plt.tight_layout()

        if save_path and self.config.save_visualizations:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


if __name__ == "__main__":
    test_trajectory_drawing()
