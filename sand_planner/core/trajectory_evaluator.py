#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 的轨迹评估器。
Trajectory evaluator for SanD-planner.

重要：create_esdf_query() 方法定义了若干嵌套闭包函数
(baselink_to_camera, trilinear_interpolate, trilinear_interpolate_batch,
esdf_query, esdf_query_batch)，它们作为闭包被原样保留。
CRITICAL: The create_esdf_query() method defines nested closure functions
(baselink_to_camera, trilinear_interpolate, trilinear_interpolate_batch,
esdf_query, esdf_query_batch). These are preserved verbatim as closures.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from sand_planner.config import InferenceConfig
from sand_planner.utils.esdf import quick_depth_to_esdf
from sand_planner.trajectory.evaluation_vectorized import evaluate_trajectories_vectorized
from sand_planner.trajectory.evaluation import TrajectoryEvaluator as TrajEval

class TrajectoryEvaluator:
    """轨迹评估器。 / Trajectory evaluator."""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def reset_mapper(self):
        """轻量级重置：清空 CUDA 缓存并执行垃圾回收。
        Lightweight reset: clear the CUDA cache and run garbage collection.
        """
        # CUDA 垃圾回收 / CUDA garbage collection.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Python 垃圾回收 / Python garbage collection.
        import gc
        gc.collect()

        if self.config.show_verbose:
            print("🔄 轻量级重置：垃圾回收完成 / Lightweight reset: GC done")

    def create_esdf_query(self, depth_img: np.ndarray):
        """创建 ESDF 查询函数。 / Create the ESDF query function."""

        # 将归一化的深度图转换为米单位（depth_img 是已经归一化到 [0,1] 的数据）
        # Convert the normalized depth image to meters (depth_img is already normalized to [0,1]).
        depth_meters = depth_img * self.config.max_depth

        try:
            # 使用 CPU/CuPy (EDT) 方法生成 ESDF
            # Generate the ESDF with the CPU/CuPy (EDT) method.
            if self.config.show_verbose:
                print("📊 使用 CPU 方法生成 ESDF / Generating ESDF with the CPU method...")
            esdf, metadata = quick_depth_to_esdf(
                depth_meters,
                self.config.camera_intrinsics,
                **self.config.esdf_config
            )

            # 坐标变换函数 / Coordinate transform functions.
            def baselink_to_camera(p_bl):
                """baselink 到 camera 坐标系变换。 / Transform from the baselink frame to the camera frame."""
                p_bl = np.asarray(p_bl, dtype=np.float64)
                t_bc = np.array([0.0, 0.0, 0.10], dtype=np.float64)
                p_cam_link = p_bl - t_bc

                x_cam = -p_cam_link[1]  # 相机 X(右) = -baselink Y(左) / camera X(right) = -baselink Y(left)
                y_cam = -p_cam_link[2]  # 相机 Y(下) = -baselink Z(上) / camera Y(down) = -baselink Z(up)
                z_cam = p_cam_link[0]   # 相机 Z(前) = baselink X(前) / camera Z(forward) = baselink X(forward)

                return np.array([x_cam, y_cam, z_cam], dtype=np.float64)

            def trilinear_interpolate(esdf_data, query_point, voxel_size, grid_origin):
                """三线性插值查询。 / Trilinear interpolation query."""
                voxel_coords = (query_point - grid_origin) / voxel_size
                i0, j0, k0 = np.floor(voxel_coords).astype(int)
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

                if (i0 < 0 or i1 >= esdf_data.shape[0] or
                    j0 < 0 or j1 >= esdf_data.shape[1] or
                    k0 < 0 or k1 >= esdf_data.shape[2]):
                    return None

                di = voxel_coords[0] - i0
                dj = voxel_coords[1] - j0
                dk = voxel_coords[2] - k0

                # 获取 8 个角点的 ESDF 值并进行三线性插值
                # Fetch the ESDF values at the 8 corner points and perform trilinear interpolation.
                v000 = esdf_data[i0, j0, k0]
                v001 = esdf_data[i0, j0, k1]
                v010 = esdf_data[i0, j1, k0]
                v011 = esdf_data[i0, j1, k1]
                v100 = esdf_data[i1, j0, k0]
                v101 = esdf_data[i1, j0, k1]
                v110 = esdf_data[i1, j1, k0]
                v111 = esdf_data[i1, j1, k1]

                v00 = v000 * (1 - di) + v100 * di
                v01 = v001 * (1 - di) + v101 * di
                v10 = v010 * (1 - di) + v110 * di
                v11 = v011 * (1 - di) + v111 * di

                v0 = v00 * (1 - dj) + v10 * dj
                v1 = v01 * (1 - dj) + v11 * dj

                result = v0 * (1 - dk) + v1 * dk
                return float(result)

            def esdf_query(point_3d):
                """单点查询函数。 / Single-point query function."""
                x, y, z = point_3d
                if self.config.clearance_height is not None:
                    z = self.config.clearance_height
                point_cam = baselink_to_camera([x, y, z])

                distance = trilinear_interpolate(
                    esdf, point_cam,
                    metadata['voxel_size'],
                    metadata['grid_origin']
                )

                if distance is not None:
                    return float(distance)
                else:
                    z_forward = max(0.5, point_cam[2])
                    return max(1.0, z_forward * 0.5)

            def trilinear_interpolate_batch(esdf_data, query_points, voxel_size, grid_origin):
                """批量三线性插值查询（完全向量化）。 / Batched trilinear interpolation query (fully vectorized).

                批量版本的三线性插值，对 N 个查询点一次性完成边界检查与插值。
                Batched version of trilinear interpolation that performs boundary checking and
                interpolation for N query points at once.

                Args:
                    esdf_data: (D, H, W) ESDF 网格 / (D, H, W) ESDF grid.
                    query_points: (N, 3) 查询点 / (N, 3) query points.
                    voxel_size: 体素大小 / voxel size.
                    grid_origin: 网格原点 / grid origin.

                Returns:
                    results: (N,) ESDF 距离，越界（None）的位置用 np.nan 表示 / (N,) ESDF distances; out-of-range (None) positions are marked as np.nan.
                """
                N = len(query_points)

                # 1. 批量计算体素坐标 / Compute voxel coordinates in batch.
                voxel_coords = (query_points - grid_origin) / voxel_size  # (N, 3)
                i0 = np.floor(voxel_coords[:, 0]).astype(int)
                j0 = np.floor(voxel_coords[:, 1]).astype(int)
                k0 = np.floor(voxel_coords[:, 2]).astype(int)
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

                # 2. 批量边界检查 / Boundary check in batch.
                valid = (
                    (i0 >= 0) & (i1 < esdf_data.shape[0]) &
                    (j0 >= 0) & (j1 < esdf_data.shape[1]) &
                    (k0 >= 0) & (k1 < esdf_data.shape[2])
                )

                # 3. 初始化结果 / Initialize the results.
                results = np.full(N, np.nan, dtype=np.float64)

                if not np.any(valid):
                    return results

                # 4. 只对有效点进行插值（向量化） / Interpolate only the valid points (vectorized).
                valid_idx = np.where(valid)[0]
                i0_v, j0_v, k0_v = i0[valid], j0[valid], k0[valid]
                i1_v, j1_v, k1_v = i1[valid], j1[valid], k1[valid]

                # 插值权重 / Interpolation weights.
                di = voxel_coords[valid, 0] - i0_v
                dj = voxel_coords[valid, 1] - j0_v
                dk = voxel_coords[valid, 2] - k0_v

                # 批量获取 8 个角点的 ESDF 值 / Fetch the ESDF values at the 8 corner points in batch.
                v000 = esdf_data[i0_v, j0_v, k0_v]
                v001 = esdf_data[i0_v, j0_v, k1_v]
                v010 = esdf_data[i0_v, j1_v, k0_v]
                v011 = esdf_data[i0_v, j1_v, k1_v]
                v100 = esdf_data[i1_v, j0_v, k0_v]
                v101 = esdf_data[i1_v, j0_v, k1_v]
                v110 = esdf_data[i1_v, j1_v, k0_v]
                v111 = esdf_data[i1_v, j1_v, k1_v]

                # 批量三线性插值 / Batched trilinear interpolation.
                v00 = v000 * (1 - di) + v100 * di
                v01 = v001 * (1 - di) + v101 * di
                v10 = v010 * (1 - di) + v110 * di
                v11 = v011 * (1 - di) + v111 * di

                v0 = v00 * (1 - dj) + v10 * dj
                v1 = v01 * (1 - dj) + v11 * dj

                results[valid] = v0 * (1 - dk) + v1 * dk

                return results

            def esdf_query_batch(points_3d):
                """批量查询函数 - 完全向量化版本。 / Batched query function - fully vectorized version.

                优化策略：
                1. 批量坐标转换；
                2. 批量三线性插值（避免 for 循环）；
                3. 向量化默认值处理。
                预期加速：5-10 倍。

                Optimization strategy:
                1. Batched coordinate transformation.
                2. Batched trilinear interpolation (avoids a for loop).
                3. Vectorized default-value handling.
                Expected speedup: 5-10x.

                Args:
                    points_3d: (N, 3) numpy 数组，N 个 3D 点 / (N, 3) numpy array of N 3D points.

                Returns:
                    distances: (N,) numpy 数组，每个点的 ESDF 距离 / (N,) numpy array of the ESDF distance for each point.
                """
                points_3d = np.asarray(points_3d, dtype=np.float64)
                if len(points_3d.shape) == 1:
                    points_3d = points_3d.reshape(1, -1)

                N = len(points_3d)

                # 1. 批量设置 clearance 高度 / Set the clearance height in batch.
                original_z = points_3d[:, 2].copy() if self.config.clearance_height is not None else None
                if self.config.clearance_height is not None:
                    points_3d = points_3d.copy()
                    points_3d[:, 2] = self.config.clearance_height

                # 2. 批量坐标转换（与单点转换 baselink_to_camera 保持一致）：baselink -> camera，先减去偏移，再做坐标轴转换。
                # Batched coordinate transformation (consistent with the single-point baselink_to_camera): baselink -> camera, first subtract the offset, then perform the axis transformation.
                t_bc = np.array([0.0, 0.0, 0.10], dtype=np.float64)
                points_cam_link = points_3d - t_bc  # 先减去偏移 / Subtract the offset first.

                # 坐标轴转换 / Axis transformation.
                points_cam = np.zeros_like(points_3d)
                points_cam[:, 0] = -points_cam_link[:, 1]  # x_cam = -y_bl
                points_cam[:, 1] = -points_cam_link[:, 2]  # y_cam = -z_bl
                points_cam[:, 2] = points_cam_link[:, 0]   # z_cam = x_bl

                if N > 0 and hasattr(self, '_debug_query_count'):
                    self._debug_query_count += 1

                # 3. 批量 ESDF 插值（完全向量化！） / Batched ESDF interpolation (fully vectorized!).
                distances = trilinear_interpolate_batch(
                    esdf, points_cam,
                    metadata['voxel_size'],
                    metadata['grid_origin']
                )

                # 4. 向量化处理无效值 / Handle invalid values in a vectorized way.
                invalid_mask = np.isnan(distances)
                if np.any(invalid_mask):
                    z_forward = np.maximum(0.5, points_cam[invalid_mask, 2])
                    distances[invalid_mask] = np.maximum(1.0, z_forward * 0.5)


                return distances

            # 初始化调试计数器 / Initialize the debug counter.
            self._debug_query_count = 0

            # 为兼容性，将批量查询函数附加到单点查询函数上
            # For compatibility, attach the batched query function to the single-point query function.
            esdf_query.batch = esdf_query_batch

            return esdf_query

        except Exception as e:
            if self.config.show_verbose:
                print(f"ESDF生成失败: {e}")
            return None

    def evaluate_trajectories(self, trajectories: List[np.ndarray], target_goal: np.ndarray, esdf_query_fn, clearance_max_points: int = 10) -> Tuple[int, Dict]:
        """评估轨迹并返回最佳索引（使用向量化优化）。 / Evaluate trajectories and return the best index (using a vectorized optimization)."""
        if esdf_query_fn is None:
            return 0, {}

        try:
            # 尝试使用向量化版本（如果存在 batch 函数）
            # Try the vectorized version (if a batch function is available).
            if hasattr(esdf_query_fn, 'batch'):
                # 使用向量化批量评估 / Use vectorized batched evaluation.
                results = evaluate_trajectories_vectorized(
                    trajectories=trajectories,
                    goal_point=target_goal,
                    esdf_query_batch_fn=esdf_query_fn.batch,
                    weights={'clear': self.config.eval_weight_clear, 'length': self.config.eval_weight_length, 'goal': self.config.eval_weight_goal},
                    safety_margin=self.config.eval_safety_margin,
                    clearance_max_points=clearance_max_points
                )


            else:
                print('wrong esdf_query_fn, no batch function!')
                # 回退到原始（非向量化）版本：
                # Fall back to the original (non-vectorized) version:
                # evaluator = TrajEval(
                #     esdf_query_function=esdf_query_fn,
                #     safety_margin=0.15,
                #     weights={'clear': 100000.0, 'length': 0.01, 'goal': 0.0001}
                # )

                # results = evaluator.evaluate_trajectories(
                #     trajectories, target_goal, return_ranking=True, clearance_max_points=clearance_max_points
                # )

            # 安全检查结果 / Sanity-check the results.
            if results and 'results' in results and len(results['results']) > 0:
                best_index = results['results'][0]['trajectory_id']
            else:
                if self.config.show_verbose:
                    print("⚠️ 警告: 轨迹评估结果为空，使用默认索引 0")
                best_index = 0
                if not results:
                    results = {'results': []}

            return best_index, results

        except Exception as e:
            if self.config.show_verbose:
                print(f"轨迹评估失败: {e}")
            return 0, {}

    def _debug_esdf_queries(self, trajectories: List[np.ndarray], esdf_query_fn, results: Dict):
        """调试 ESDF 查询，打印每条轨迹的 ESDF 距离详情。 / Debug ESDF queries by printing the ESDF distance details of each trajectory."""
        print(f"\n{'='*80}")
        print(f"🔍 ESDF查询调试")
        print(f"{'='*80}")

        # 只检查前 10 条和后 10 条轨迹，以及最佳轨迹
        # Only inspect the first and last 10 trajectories, plus the best trajectory.
        num_trajs = len(trajectories)
        best_idx = results['results'][0]['trajectory_id'] if results and 'results' in results and len(results['results']) > 0 else 0

        # 选择要检查的轨迹索引 / Select the trajectory indices to inspect.
        check_indices = set()
        check_indices.add(best_idx)  # 最佳轨迹 / The best trajectory.
        check_indices.update(range(min(5, num_trajs)))  # 前 5 条 / The first 5.
        check_indices.update(range(max(0, num_trajs-5), num_trajs))  # 后 5 条 / The last 5.

        # 如果最佳轨迹不在前 5 或后 5 中，也检查它周围的轨迹
        # If the best trajectory is not among the first or last 5, also inspect the trajectories around it.
        if best_idx >= 5 and best_idx < num_trajs - 5:
            check_indices.update(range(max(0, best_idx-2), min(num_trajs, best_idx+3)))

        check_indices = sorted(list(check_indices))

        print(f"检查轨迹: {check_indices}")
        print(f"最佳轨迹索引: {best_idx}")
        print(f"安全阈值: 0.45m\n")

        for idx in check_indices:
            if idx >= len(trajectories):
                continue

            traj = trajectories[idx]

            # 查询 ESDF 距离 / Query the ESDF distance.
            if hasattr(esdf_query_fn, 'batch'):
                distances = esdf_query_fn.batch(traj)
            else:
                distances = np.array([esdf_query_fn(p) for p in traj])

            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)

            # 碰撞检测 / Collision detection.
            num_unsafe = np.sum(distances < 0.45)
            is_safe = num_unsafe == 0

            # 找到对应的结果 / Find the corresponding result.
            traj_result = None
            if results and 'results' in results:
                for r in results['results']:
                    if r['trajectory_id'] == idx:
                        traj_result = r
                        break

            # 打印标记 / Print marker.
            marker = "🟢 [最佳]" if idx == best_idx else ("✅" if is_safe else "❌")

            print(f"{marker} 轨迹 #{idx:3d}:")
            print(f"   轨迹点数: {len(traj)}")
            print(f"   终点位置: [{traj[-1][0]:6.2f}, {traj[-1][1]:6.2f}, {traj[-1][2]:6.2f}]")
            print(f"   ESDF距离: min={min_dist:.3f}m, max={max_dist:.3f}m, mean={mean_dist:.3f}m")
            print(f"   不安全点: {num_unsafe}/{len(traj)} ({num_unsafe/len(traj)*100:.1f}%)")

            if traj_result:
                print(f"   评估分数: {traj_result.get('total_cost', 0):.2f}")
                print(f"   安全得分: {traj_result.get('clearance_cost', 0):.2f}")

            # 打印轨迹各段的详细 ESDF（前、中、后）
            # Print detailed ESDF values along the trajectory segments (front, middle, back).
            print(f"   轨迹ESDF采样点:")
            sample_indices = [0, len(traj)//4, len(traj)//2, len(traj)*3//4, len(traj)-1]
            for idx in sample_indices:
                if idx < len(traj):
                    p = traj[idx]
                    d = distances[idx]
                    status = "✅" if d >= 0.45 else "❌"
                    print(f"      P{idx:2d}: [{p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f}] → {d:.3f}m {status}")

            print()

        print(f"{'='*80}\n")
