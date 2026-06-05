#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 推理流水线的编排器 / Orchestrator for the SanD-planner inference pipeline.

负责串联深度处理、模型推理、轨迹采样与评估等各组件。
Wires together depth processing, model inference, trajectory sampling, and evaluation.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sand_planner.config import InferenceConfig
from sand_planner.agent.depth_processor import FileDepthProcessor
from sand_planner.core.model_manager import ModelManager
from sand_planner.core.trajectory_inference import TrajectoryInference
from sand_planner.core.trajectory_evaluator import TrajectoryEvaluator
from sand_planner.trajectory.arc_length_sampling_vectorized import sample_predicted_trajectories_vectorized


class SandPlannerInference:
    """主推理类 / Main inference class."""

    def __init__(self, config: InferenceConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.depth_processor = FileDepthProcessor(config)
        self.model_manager = ModelManager(config)

        # 从轨迹可视化模块导入 Visualizer
        # Import Visualizer from the trajectory visualization module
        from sand_planner.trajectory.visualization import Visualizer
        self.visualizer = Visualizer(config)
        self.evaluator = TrajectoryEvaluator(config)

        # 初始化组件 / Initialize components
        self.model, self.normalizer = self.model_manager.load_model()
        self.inference_engine = TrajectoryInference(config, self.model, self.normalizer)

        # 由 Agent 传入的原始深度图（用于 ESDF 查询）
        # Original depth image passed in from the Agent (used for ESDF queries)
        self.agent_original_depth: Optional[np.ndarray] = None

        # 推理计数器（用于周期性深度清理）
        # Inference counter (used for periodic deep cleanup)
        self._inference_count = 0
        self._deep_cleanup_interval = 10  # 每 10 次推理做一次深度清理 / deep cleanup every 10 inferences

    def _compute_remaining_trajectory(self, trajectory: np.ndarray, execution_distance: float = 0.2) -> np.ndarray:
        """计算轨迹剩余部分（去除已执行的点） / Compute the remaining part of a trajectory (drop already-executed points).

        Args:
            trajectory: (N, 3) 完整轨迹 / full trajectory.
            execution_distance: 已执行的距离（米） / distance already executed (meters).

        Returns:
            remaining_trajectory: (M, 3) 剩余轨迹，M <= N / remaining trajectory, with M <= N.
        """
        if len(trajectory) == 0:
            return trajectory

        # 计算累积弧长距离 / Compute cumulative arc-length distances
        diffs = np.diff(trajectory, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])

        # 找到第一个超过 execution_distance 的点
        # Find the first point past execution_distance
        start_idx = np.searchsorted(cumulative_distances, execution_distance)

        if start_idx >= len(trajectory):
            return np.array([])  # 整条轨迹都已执行 / the whole trajectory has been executed

        return trajectory[start_idx:]

    def _apply_backtracking_bonus(self, evaluation_results: Dict, prev_traj_index: int):
        """给上一帧最优轨迹添加奖励分数（实现迟滞效应） / Add a bonus score to the previous frame's best trajectory (hysteresis effect).

        Args:
            evaluation_results: 评估结果字典 / evaluation results dictionary.
            prev_traj_index: 上一帧轨迹在候选池中的索引 / index of the previous trajectory in the candidate pool.
        """
        if 'results' not in evaluation_results or not evaluation_results['results']:
            return

        # 找到旧轨迹的评估结果 / Find the evaluation result of the old trajectory
        old_traj_result = None
        for result in evaluation_results['results']:
            if result['trajectory_id'] == prev_traj_index:
                old_traj_result = result
                break

        if old_traj_result is None:
            return

        # 计算奖励分数（基于配置的 bonus 系数）
        # Compute the bonus score (based on the configured bonus coefficient)
        bonus = self.inference_engine._backtracking_bonus
        original_score = old_traj_result.get('final_score', 0.0)

        # 添加奖励（相对于原分数的百分比）
        # Add the bonus (as a percentage of the original score)
        old_traj_result['final_score'] = original_score * (1 + bonus)
        old_traj_result['backtracking_bonus'] = bonus

        if self.verbose:
            print(f"🎁 [Backtracking] 旧轨迹奖励: {original_score:.4f} → {old_traj_result['final_score']:.4f} (+{bonus*100:.1f}%)")

    def reset_environment(self):
        """重置环境状态，清理 CUDA 内存 / Reset the environment state and free CUDA memory.

        在每个 episode 结束时调用，防止内存泄漏。
        Called at the end of each episode to prevent memory leaks.
        """
        # 清理评估器缓存与显存 / Reset evaluator caches and free GPU memory
        self.evaluator.reset_mapper()
        self.inference_engine.reset_warm_start_cache()

        # 重置计数器 / Reset the counter
        self._inference_count = 0

        # 强制 CUDA 同步并执行垃圾回收 / Force CUDA synchronization and garbage collection
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if self.verbose:
            print("🔄 环境已重置，CUDA内存已清理")

    def process_single_depth(self, depth_file: str) -> Dict[str, Any]:
        """处理单个深度图像 / Process a single depth image."""
        depth_path = os.path.join(self.config.input_depth_dir, depth_file)

        # 记录各个步骤的时间 / Record the time of each step
        step_timing = {}

        # 加载深度图像 / Load the depth image
        load_start = time.time()
        if self.config.use_consecutive_frames and self.config.sequence_length > 1:
            if os.path.exists(self.config.consecutive_depth_dir):
                depth_sequence = self.depth_processor.load_consecutive_sequence()
                depth_for_viz = self.depth_processor.load_for_visualization(
                    os.path.join(self.config.consecutive_depth_dir, f"depth_{self.config.sequence_length:04d}.png")
                )
            else:
                depth_for_model = self.depth_processor.load_for_model(depth_path)
                depth_sequence = self.depth_processor.create_sequence(depth_for_model)
                depth_for_viz = self.depth_processor.load_for_visualization(depth_path)
        else:
            depth_for_model = self.depth_processor.load_for_model(depth_path)
            depth_sequence = self.depth_processor.create_sequence(depth_for_model)
            depth_for_viz = self.depth_processor.load_for_visualization(depth_path)
        step_timing['depth_loading'] = time.time() - load_start

        # 生成轨迹 / Generate trajectories
        inference_start = time.time()
        depth_batch = depth_sequence.unsqueeze(0).to(self.config.device)
        control_points, timing = self.inference_engine.generate_trajectories(depth_batch, self.config.target_position)
        step_timing['trajectory_generation'] = time.time() - inference_start

        # 轨迹采样 - 根据 prediction_mode 决定是否进行样条重建
        # waypoints 模式: 直接返回预测的 8 个 waypoints (0.2m 间隔)
        # control_points 模式: 样条拟合后等弧长采样
        # Trajectory sampling - prediction_mode decides whether spline reconstruction is performed:
        #   waypoints mode: directly return the 8 predicted waypoints (0.2 m spacing)
        #   control_points mode: fit a B-spline, then resample at equal arc length
        sampling_start = time.time()
        try:
            # 将 (B,8,3) 转为 list of (8,3) / Convert (B,8,3) into a list of (8,3)
            control_points_list = [control_points[i] for i in range(control_points.shape[0])]
            sampled_trajectories = sample_predicted_trajectories_vectorized(
                control_points_list,
                arc_length=self.config.arc_length_step,  # 仅在 control_points 模式生效 / only effective in control_points mode
                method=self.config.trajectory_interpolation,
                prediction_mode=self.config.prediction_mode
            )
        except Exception:
            sampled_trajectories = None
        step_timing['arc_length_sampling'] = time.time() - sampling_start

        # 轨迹评估 / Trajectory evaluation
        eval_start = time.time()
        best_index = 0
        evaluation_results = {}
        if sampled_trajectories:
            # Best Plan Candidate Backtracking: 将上一帧最优轨迹加入候选池
            # Best Plan Candidate Backtracking: add the previous frame's best trajectory into the candidate pool
            trajectories_to_evaluate = sampled_trajectories.copy()
            prev_traj_index = None  # 追踪旧轨迹在候选池中的位置 / track the old trajectory's position in the candidate pool

            if self.inference_engine._enable_plan_backtracking and self.inference_engine._prev_best_trajectory is not None:
                # 计算剩余轨迹（去除已执行部分） / Compute the remaining trajectory (drop the executed part)
                prev_traj = self.inference_engine._prev_best_trajectory
                exec_dist = self.config.execution_distance_per_frame
                remaining_traj = self._compute_remaining_trajectory(prev_traj, execution_distance=exec_dist)

                if len(remaining_traj) >= 10:  # 只有剩余足够长才加入 / only add it if enough length remains
                    trajectories_to_evaluate.append(remaining_traj)
                    prev_traj_index = len(trajectories_to_evaluate) - 1
                    if self.verbose:
                        print(f"🔄 [Backtracking] 添加上一帧最优轨迹剩余部分 (索引{prev_traj_index}, 剩余{len(remaining_traj)}点)")

            # 直接使用内存中的 depth_for_viz 数据生成 ESDF（保持原始分辨率）
            # Build the ESDF directly from the in-memory depth_for_viz data (keeping original resolution)
            esdf_query_fn = self.evaluator.create_esdf_query(depth_for_viz)
            best_index, evaluation_results = self.evaluator.evaluate_trajectories(
                trajectories_to_evaluate, np.array(self.config.target_position), esdf_query_fn, clearance_max_points=self.config.clearance_max_points
            )

            # 如果选中了旧轨迹，给予奖励加分（实现迟滞效应）
            # If the old trajectory is selected, apply a bonus score (hysteresis effect)
            if prev_traj_index is not None and evaluation_results.get('results'):
                self._apply_backtracking_bonus(evaluation_results, prev_traj_index)
                # 重新排序以获取最佳索引 / Re-sort to obtain the best index
                best_result = max(evaluation_results['results'], key=lambda x: x.get('final_score', -float('inf')))
                best_index = best_result['trajectory_id']

            # 更新最优轨迹缓存（用于下一帧） / Update the best-trajectory cache (for the next frame)
            if best_index < len(sampled_trajectories):
                # 新生成的轨迹 / A newly generated trajectory
                self.inference_engine._prev_best_trajectory = sampled_trajectories[best_index].copy()
                self.inference_engine._prev_best_control_points = control_points[best_index].copy()

                # 更新 initial_turn 缓存：计算 CP1 和 CP2 的 Y 坐标均值（归一化空间）
                # Update the initial_turn cache: average Y of CP1 and CP2 (in normalized space)
                if hasattr(self.inference_engine.model.condition_encoder, 'use_initial_turn') and \
                   self.inference_engine.model.condition_encoder.use_initial_turn:
                    best_cp = control_points[best_index]  # (8, 3)
                    # 在归一化空间中计算 / Compute in normalized space
                    if self.inference_engine.normalizer is not None:
                        # 将控制点归一化 / Normalize the control points
                        cp_norm = self.inference_engine.normalizer.normalize(best_cp)
                        # 计算 CP1 和 CP2 的 Y 均值 / Average Y of CP1 and CP2
                        y_avg = (cp_norm[1, 1] + cp_norm[2, 1]) / 2.0
                        # 备选/alt: y_avg = cp_norm[1, 1]
                    else:
                        # 无归一化时直接使用原始值 / Without normalization, use raw values directly
                        y_avg = (best_cp[1, 1] + best_cp[2, 1]) / 2.0
                    self.inference_engine._prev_initial_turn = float(y_avg)
            elif prev_traj_index is not None and best_index == prev_traj_index:
                # 保持旧轨迹（已经在缓存中） / Keep the old trajectory (already in the cache)
                if self.verbose:
                    print(f"✅ [Backtracking] 保持上一帧轨迹（迟滞效应）")
                # initial_turn 缓存保持不变 / the initial_turn cache stays unchanged

        self.inference_engine.update_warm_start_cache(best_index)
        step_timing['trajectory_evaluation'] = time.time() - eval_start

        # 可视化 / Visualization
        viz_start = time.time()
        base_name = os.path.splitext(depth_file)[0]
        save_path = os.path.join(self.config.output_dir, f"{base_name}_result.png")
        fig = self.visualizer.visualize_trajectories(depth_for_viz, control_points, best_index, save_path)
        plt.close(fig)
        step_timing['visualization'] = time.time() - viz_start

        # 保存数据 / Save data
        save_start = time.time()
        if self.config.save_data:
            self._save_results(base_name, control_points, sampled_trajectories, best_index, timing, evaluation_results)
        step_timing['data_saving'] = time.time() - save_start

        # 合并时间统计 / Merge timing statistics
        timing.update(step_timing)

        return {
            'control_points': control_points,
            'sampled_trajectories': sampled_trajectories,
            'best_index': best_index,
            'timing': timing,
            'evaluation_results': evaluation_results
        }

    def _save_results(self, base_name: str, control_points: np.ndarray,
                     sampled_trajectories: Optional[List], best_index: int,
                     timing: Dict, evaluation_results: Dict):
        """保存结果数据 / Save result data."""
        # 保存控制点 / Save control points
        np.save(os.path.join(self.config.output_dir, f"{base_name}_control_points.npy"), control_points)

        # 保存采样轨迹 / Save sampled trajectories
        if sampled_trajectories:
            np.save(os.path.join(self.config.output_dir, f"{base_name}_trajectories.npy"),
                   np.array(sampled_trajectories, dtype=object))

        # 保存信息 / Save metadata
        info = {
            'target_position': self.config.target_position,
            'total_trajectories': len(control_points),
            'best_trajectory_index': best_index,
            'sequence_length': self.config.sequence_length,
            'fusion_strategy': self.config.fusion_strategy,
            'timing': timing,
            'evaluation_summary': evaluation_results.get('summary', {})
        }

        with open(os.path.join(self.config.output_dir, f"{base_name}_info.json"), 'w') as f:
            json.dump(info, f, indent=2)

    def run_inference(self, depth_file: Optional[str] = None) -> Dict[str, Any]:
        """运行推理 / Run inference."""
        if depth_file is None:
            # 使用目录中第一个文件 / Use the first file in the directory
            depth_files = sorted([f for f in os.listdir(self.config.input_depth_dir) if f.endswith('.png')])
            if not depth_files:
                raise ValueError(f"未找到深度图像文件: {self.config.input_depth_dir}")
            depth_file = depth_files[0]

        if self.verbose:
            print(f"开始推理: {depth_file}")
            print(f"配置: seq_len={self.config.sequence_length}, fusion={self.config.fusion_strategy}")
            print(f"目标位置: {self.config.target_position}")

        start_time = time.time()
        results = self.process_single_depth(depth_file)
        total_time = time.time() - start_time

        # 详细时间统计 / Detailed timing statistics
        timing = results['timing']
        if self.verbose:
            print(f"\n⏱️ 完整流程时间分析:")
            print(f"深度图加载: {timing.get('depth_loading', 0):.3f}秒")
            print(f"轨迹生成: {timing.get('trajectory_generation', 0):.3f}秒")
            print(f"  ├─ 调度器初始化: {timing.get('scheduler_init', 0):.3f}秒")
            print(f"  ├─ 条件编码: {timing.get('condition_encoding', 0):.3f}秒")
            print(f"  ├─ 噪声初始化: {timing.get('noise_init', 0):.3f}秒")
            print(f"  ├─ DDPM采样: {timing.get('sampling', 0):.3f}秒")
            print(f"  └─ 后处理: {timing.get('post_processing', 0):.3f}秒")
            print(f"等弧长采样: {timing.get('arc_length_sampling', 0):.3f}秒")
            print(f"轨迹评估: {timing.get('trajectory_evaluation', 0):.3f}秒")
            print(f"可视化保存: {timing.get('visualization', 0):.3f}秒")
            print(f"数据保存: {timing.get('data_saving', 0):.3f}秒")
            print(f"总时间: {total_time:.3f}秒")

        # 计算各部分占比 / Compute each part's percentage share
        if total_time > 0 and self.verbose:
            print(f"\n📊 时间占比分析:")
            print(f"深度图加载: {timing.get('depth_loading', 0)/total_time*100:.1f}%")
            print(f"轨迹生成: {timing.get('trajectory_generation', 0)/total_time*100:.1f}%")
            print(f"  ├─ 调度器: {timing.get('scheduler_init', 0)/total_time*100:.1f}%")
            print(f"  ├─ 条件编码: {timing.get('condition_encoding', 0)/total_time*100:.1f}%")
            print(f"  ├─ 噪声初始化: {timing.get('noise_init', 0)/total_time*100:.1f}%")
            print(f"  ├─ DDMP采样: {timing.get('sampling', 0)/total_time*100:.1f}%")
            print(f"  └─ 后处理: {timing.get('post_processing', 0)/total_time*100:.1f}%")
            print(f"等弧长采样: {timing.get('arc_length_sampling', 0)/total_time*100:.1f}%")
            print(f"轨迹评估: {timing.get('trajectory_evaluation', 0)/total_time*100:.1f}%")
            print(f"可视化保存: {timing.get('visualization', 0)/total_time*100:.1f}%")
            print(f"数据保存: {timing.get('data_saving', 0)/total_time*100:.1f}%")

        if self.verbose:
            print(f"最佳轨迹索引: {results['best_index']}")
            print(f"输出目录: {self.config.output_dir}")

        return results

    def process_depth_arrays(self, depth_sequences: torch.Tensor) -> Dict[str, Any]:
        """直接处理深度序列数组，避免文件 IO / Process depth-sequence arrays directly, avoiding file IO.

        Args:
            depth_sequences: (batch_size, seq_len, 1, H, W) 深度序列张量 / depth-sequence tensor.

        Returns:
            Dict[str, Any]: 推理结果字典 / inference results dictionary.
        """
        step_timing = {}

        if self.verbose:
            print(f"🔄 直接处理深度序列: {depth_sequences.shape}")

        # 确保在正确的设备上 / Make sure it is on the correct device
        depth_sequences = depth_sequences.to(self.config.device)

        # 取第一个 batch 进行推理（与原逻辑保持一致）
        # Take the first batch for inference (consistent with the original logic)
        depth_batch = depth_sequences[0:1]  # (1, seq_len, 1, H, W)

        # 为可视化准备深度数据 / Prepare depth data for visualization
        viz_depth = depth_sequences[0, -1, 0].cpu().numpy()  # 取最后一帧用于可视化 (H, W) / use the last frame for visualization

        # 尝试获取原始分辨率深度用于 ESDF（如果 agent 提供的话）
        # Try to obtain the original-resolution depth for the ESDF (if the agent provides it)
        original_depth_for_esdf = None
        if hasattr(self, 'agent_original_depth') and self.agent_original_depth is not None:
            original_depth_for_esdf = self.agent_original_depth
            if self.verbose:
                print(f"📊 使用原始深度构建ESDF: shape={original_depth_for_esdf.shape}, range=[{original_depth_for_esdf.min():.3f}, {original_depth_for_esdf.max():.3f}]")
            # 使用后立即清除，避免下次推理时误用旧数据
            # Clear it right after use to avoid mistakenly reusing stale data next inference
            self.agent_original_depth = None
        else:
            original_depth_for_esdf = viz_depth
            if self.verbose:
                print(f"⚠️ 降级使用下采样深度构建ESDF: shape={viz_depth.shape}, range=[{viz_depth.min():.3f}, {viz_depth.max():.3f}]")

        # 生成轨迹 / Generate trajectories
        inference_start = time.time()
        control_points, timing = self.inference_engine.generate_trajectories(depth_batch, self.config.target_position)
        step_timing['trajectory_generation'] = time.time() - inference_start

        # 轨迹采样 - 根据 prediction_mode 决定是否进行样条重建
        # waypoints 模式: 直接返回预测的 8 个 waypoints (0.2m 间隔)
        # control_points 模式: 样条拟合后等弧长采样
        # Trajectory sampling - prediction_mode decides whether spline reconstruction is performed:
        #   waypoints mode: directly return the 8 predicted waypoints (0.2 m spacing)
        #   control_points mode: fit a B-spline, then resample at equal arc length
        sampling_start = time.time()
        try:
            # 将 (B,8,3) 转为 list of (8,3) / Convert (B,8,3) into a list of (8,3)
            control_points_list = [control_points[i] for i in range(control_points.shape[0])]
            sampled_trajectories = sample_predicted_trajectories_vectorized(
                control_points_list,
                arc_length=self.config.arc_length_step,  # 仅在 control_points 模式生效 / only effective in control_points mode
                method=self.config.trajectory_interpolation,
                prediction_mode=self.config.prediction_mode
            )
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 采样失败: {e}")
            sampled_trajectories = None
        step_timing['arc_length_sampling'] = time.time() - sampling_start

        # 轨迹评估 / Trajectory evaluation
        eval_start = time.time()
        best_index = 0
        evaluation_results = {}
        if sampled_trajectories:
            # 使用原始分辨率深度数据生成 ESDF（获得更高精度）
            # Build the ESDF from original-resolution depth data (for higher accuracy)
            esdf_query_fn = self.evaluator.create_esdf_query(original_depth_for_esdf)
            best_index, evaluation_results = self.evaluator.evaluate_trajectories(
                sampled_trajectories, np.array(self.config.target_position), esdf_query_fn, clearance_max_points=self.config.clearance_max_points
            )
        self.inference_engine.update_warm_start_cache(best_index)
        step_timing['trajectory_evaluation'] = time.time() - eval_start

        self._inference_count += 1

        # 可视化（仅在启用时） / Visualization (only when enabled)
        viz_start = time.time()
        if self.config.save_visualizations:
            save_path = os.path.join(self.config.output_dir, f"direct_depth_result.png")
            fig = self.visualizer.visualize_trajectories(viz_depth, control_points, best_index, save_path)
            plt.close(fig)
        step_timing['visualization'] = time.time() - viz_start

        # 合并时间统计 / Merge timing statistics
        timing.update(step_timing)

        if self.verbose:
            print(f"\n✅ 直接深度处理完成:")
            print(f"\n📊 总体耗时统计:")
            total_time = (timing.get('trajectory_generation', 0) +
                         timing.get('arc_length_sampling', 0) +
                         timing.get('trajectory_evaluation', 0) +
                         timing.get('visualization', 0))

            print(f"   ├─ 轨迹生成:     {timing.get('trajectory_generation', 0)*1000:.2f}ms ({timing.get('trajectory_generation', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   │  ├─ 调度器初始化: {timing.get('scheduler_init', 0)*1000:.2f}ms")
            print(f"   │  ├─ 条件编码:     {timing.get('condition_encoding', 0)*1000:.2f}ms")
            print(f"   │  ├─ 噪声初始化:   {timing.get('noise_init', 0)*1000:.2f}ms")
            print(f"   │  │  ├─ 随机数生成:   {timing.get('randn_generation', 0)*1000:.2f}ms")
            print(f"   │  │  └─ 修正第一点:   {timing.get('fix_first_cp', 0)*1000:.2f}ms")
            print(f"   │  ├─ DDPM采样:     {timing.get('sampling', 0)*1000:.2f}ms")
            print(f"   │  └─ 后处理:       {timing.get('post_processing', 0)*1000:.2f}ms")
            print(f"   ├─ 等弧长采样:   {timing.get('arc_length_sampling', 0)*1000:.2f}ms ({timing.get('arc_length_sampling', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   ├─ 轨迹评估:     {timing.get('trajectory_evaluation', 0)*1000:.2f}ms ({timing.get('trajectory_evaluation', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   ├─ 可视化:       {timing.get('visualization', 0)*1000:.2f}ms ({timing.get('visualization', 0)/max(total_time, 0.001)*100:.1f}%)")
            print(f"   └─ 总计:         {total_time*1000:.2f}ms")
            print(f"\n🎯 性能瓶颈: DDPM采样 ({timing.get('sampling', 0)*1000:.2f}ms, {timing.get('sampling', 0)/max(total_time, 0.001)*100:.1f}%总时间)")
            print(f"\n💡 噪声初始化分析:")
            print(f"   - 随机数生成: {timing.get('randn_generation', 0)*1000:.2f}ms")
            print(f"   - 修正第一点: {timing.get('fix_first_cp', 0)*1000:.2f}ms")
            if timing.get('randn_generation', 0) > 0.01:  # 超过 10ms / over 10 ms
                print(f"   ⚠️  随机数生成较慢，可能是首次调用或CUDA同步问题")

        return {
            'control_points': control_points,
            'sampled_trajectories': sampled_trajectories,
            'best_index': best_index,
            'timing': timing,
            'evaluation_results': evaluation_results
        }


def create_config_from_args(args) -> InferenceConfig:
    """从命令行参数创建配置 / Create a configuration from command-line arguments."""
    config = InferenceConfig()

    # 更新配置 / Update the configuration
    if hasattr(args, 'sequence_length'):
        config.sequence_length = args.sequence_length
    if hasattr(args, 'fusion_strategy'):
        config.fusion_strategy = args.fusion_strategy
    if hasattr(args, 'use_consecutive'):
        config.use_consecutive_frames = args.use_consecutive
    if hasattr(args, 'target_x'):
        config.target_position[0] = args.target_x
    if hasattr(args, 'target_y'):
        config.target_position[1] = args.target_y
    if hasattr(args, 'target_z'):
        config.target_position[2] = args.target_z
    if hasattr(args, 'batch_size'):
        config.batch_size = args.batch_size
    if hasattr(args, 'num_steps'):
        config.num_inference_steps = args.num_steps
    if hasattr(args, 'output_dir'):
        config.output_dir = args.output_dir
    if hasattr(args, 'verbose'):
        config.show_verbose = args.verbose
    if hasattr(args, 'clearance_height') and args.clearance_height is not None:
        config.clearance_height = args.clearance_height
    if hasattr(args, 'enable_warm_start') and args.enable_warm_start:
        config.enable_warm_start = True
    if hasattr(args, 'warm_start_resume_step') and args.warm_start_resume_step is not None:
        config.warm_start_resume_step = max(0, args.warm_start_resume_step)

    return config


def main():
    parser = argparse.ArgumentParser(description='SanD-planner 推理脚本')

    # 基本配置 / Basic configuration
    parser.add_argument('--sequence_length', type=int, default=1, choices=[1, 4], help='序列长度')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'average', 'attention'], help='融合策略')
    parser.add_argument('--use_consecutive', action='store_true', help='使用连续帧')

    # 目标位置 / Target position
    parser.add_argument('--target_x', type=float, default=4.0, help='目标X坐标')
    parser.add_argument('--target_y', type=float, default=1.0, help='目标Y坐标')
    parser.add_argument('--target_z', type=float, default=0.0, help='目标Z坐标')

    # 推理参数 / Inference parameters
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_steps', type=int, default=10, help='推理步数')
    parser.add_argument('--clearance_height', type=float, default=0.5, help='ESDF 查询固定离地高度 (米)，设为负值禁用固定高度')
    parser.add_argument('--enable_warm_start', action='store_true', help='启用warm-start轨迹初始化')
    parser.add_argument('--warm_start_resume_step', type=int, default=None, help='warm-start重新加噪时使用的scheduler步编号')

    # 输出配置 / Output configuration
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--depth_file', type=str, help='指定深度图像文件')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')

    args = parser.parse_args()

    # 创建配置并运行推理 / Create the configuration and run inference
    config = create_config_from_args(args)
    if args.clearance_height is not None and args.clearance_height < 0:
        config.clearance_height = None
    inference = SandPlannerInference(config)
    results = inference.run_inference(args.depth_file)

    print("推理完成!")


if __name__ == "__main__":
    from sand_planner.config import InferenceConfig
    from sand_planner.core.orchestrator import SandPlannerInference

    config = InferenceConfig()
    config.target_position = [4.0, 1.0, 0.0]
    inference = SandPlannerInference(config)
    results = inference.run_inference()

    main()
