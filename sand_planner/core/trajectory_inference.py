#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 的轨迹推理引擎 / Trajectory inference engine for SanD-planner.

封装从深度条件到 B-spline 控制点的扩散采样推理流程。
Encapsulates the diffusion-sampling inference from depth conditions to B-spline control points.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

from sand_planner.config import InferenceConfig


class TrajectoryInference:
    """轨迹推理器 / Trajectory inference engine."""

    def __init__(self, config: InferenceConfig, model, normalizer):
        self.config = config
        self.model = model
        self.normalizer = normalizer

        # 预计算归一化零点并缓存到 GPU，避免推理时重复计算
        # Precompute the normalized zero point and cache it on GPU to avoid recomputing during inference
        self._normalized_zero_cache = None
        if self.normalizer is not None:
            zero_norm_np = self.normalizer.normalized_zero()
            self._normalized_zero_cache = torch.from_numpy(zero_norm_np).to(
                device=self.config.device,
                dtype=self.model.unet.dtype
            )
            print(f"[TrajectoryInference] 归一化零点缓存: {self._normalized_zero_cache}")

        # 固定全局种子：跨帧确定性，批内多样性
        # Fix a global seed: cross-frame determinism, intra-batch diversity
        self.global_seed = 42  # 固定种子 / fixed seed
        torch.manual_seed(self.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.global_seed)
        np.random.seed(self.global_seed)

        # warm-start 相关缓存 / warm-start related caches
        self._prev_control_points: Optional[torch.Tensor] = None  # (3, 8)
        self._latest_latent_control_points: Optional[torch.Tensor] = None
        self._warm_start_counter: int = 0
        self._warm_start_enabled = getattr(self.config, 'enable_warm_start', False)
        self._warm_start_resume_step = max(0, getattr(self.config, 'warm_start_resume_step', 0))

        # Best Plan Candidate Backtracking 相关缓存 / Best plan candidate backtracking caches
        self._prev_best_trajectory: Optional[np.ndarray] = None  # 上一帧的最优轨迹 / previous frame's best trajectory (N, 3)

        # Initial Turn 相关缓存（用于时序连贯性）/ Initial-turn caches (for temporal coherence)
        self._prev_initial_turn: Optional[float] = None  # 上一帧的初始转弯值（归一化后的 Y 分量）/ previous frame's initial turn value (normalized Y component)
        self._prev_best_control_points: Optional[np.ndarray] = None  # 上一帧的最优控制点 / previous frame's best control points (8, 3)
        self._executed_distance: float = 0.0  # 已执行的距离（用于轨迹剪裁）/ distance already executed (for trajectory trimming)
        self._enable_plan_backtracking = getattr(self.config, 'enable_plan_backtracking', False)
        self._backtracking_bonus = getattr(self.config, 'backtracking_bonus', 0.05)  # 给旧轨迹的奖励分数 / bonus score awarded to the previous trajectory

    def _get_resume_timestep(self, scheduler: DPMSolverMultistepScheduler) -> torch.Tensor:
        resume_idx = min(self._warm_start_resume_step, len(scheduler.timesteps) - 1)
        return scheduler.timesteps[resume_idx]

    def _initialize_control_points(self, batch_size: int, scheduler: DPMSolverMultistepScheduler, dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
        num_cp = getattr(self.model, 'num_control_points', getattr(self.config, 'num_control_points', 8))
        shape = (batch_size, 3, num_cp)
        device = self.config.device

        if not self._warm_start_enabled or self._prev_control_points is None:
            self._warm_start_counter = 0
            return torch.randn(shape, device=device, dtype=dtype), 0

        # 复用上一帧的完整 batch（保留多样性）
        # Reuse the previous frame's full batch (preserving diversity)
        prev = self._prev_control_points.to(device=device, dtype=dtype)

        # 处理维度：如果是单条轨迹 (3, 8)，扩展为 batch
        # Handle dimensionality: if it is a single trajectory (3, 8), expand it to a batch
        if prev.ndim == 2:
            prev = prev.unsqueeze(0)

        # 处理 batch size 不匹配的情况 / Handle batch-size mismatch
        if prev.shape[0] != batch_size:
            if prev.shape[0] == 1:
                # 上一帧只有 1 条，复制到整个 batch（退化为原逻辑）
                # Previous frame has only 1 trajectory; replicate it across the whole batch (degenerates to the original logic)
                prev = prev.repeat(batch_size, 1, 1)
            elif prev.shape[0] > batch_size:
                # 上一帧更多，随机抽样 / Previous frame has more; sample randomly
                indices = torch.randperm(prev.shape[0])[:batch_size]
                prev = prev[indices]
            else:
                # 上一帧更少，循环复用 / Previous frame has fewer; reuse cyclically
                repeat_count = (batch_size + prev.shape[0] - 1) // prev.shape[0]
                prev = prev.repeat(repeat_count, 1, 1)[:batch_size]

        # 为每个样本独立生成噪声（保持 batch 内差异）
        # Generate noise independently per sample (to keep intra-batch differences)
        # 注意：这里使用标准正态分布噪声，不进行缩放；噪声水平完全由 resume_step (timestep t) 决定
        # Note: use standard normal noise here without scaling; the noise level is determined entirely by resume_step (timestep t)
        noise = torch.randn(shape, device=device, dtype=dtype)

        # 获取 resume timestep 并加噪 / Get the resume timestep and add noise
        resume_idx = min(self._warm_start_resume_step, len(scheduler.timesteps) - 1)
        resume_t = scheduler.timesteps[resume_idx]

        if resume_t.ndim == 0:
            resume_t = resume_t.unsqueeze(0)
        timesteps = resume_t.long().to(device=device)
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.repeat(batch_size)

        resumed = scheduler.add_noise(prev, noise, timesteps)
        self._warm_start_counter += 1
        return resumed, resume_idx

    def update_warm_start_cache(self, best_index: Optional[int] = None):
        """保存最优轨迹的控制点用于下次 warm start / Save the best trajectory's control points for the next warm start.

        Args:
            best_index: 最优轨迹在 batch 中的索引；如果为 None，则保存整个 batch（兼容模式）。 / Index of the best trajectory within the batch; if None, save the entire batch (compatibility mode).
        """
        if not self._warm_start_enabled:
            return
        if self._latest_latent_control_points is None:
            return

        # 如果指定了 best_index，只保存该条轨迹；否则保存整个 batch
        # If best_index is given, save only that trajectory; otherwise save the entire batch
        if best_index is not None:
            # 只保存被选中的最优轨迹 / Save only the selected best trajectory (3, 8)
            if best_index < self._latest_latent_control_points.shape[0]:
                self._prev_control_points = self._latest_latent_control_points[best_index].detach().clone().to(device='cpu')
            else:
                # best_index 超出范围（可能是回溯的旧轨迹），不更新
                # best_index is out of range (possibly the backtracked previous trajectory); skip the update
                pass
        else:
            # 兼容模式：保存整个 batch / Compatibility mode: save the entire batch
            self._prev_control_points = self._latest_latent_control_points.detach().clone().to(device='cpu')

    def reset_warm_start_cache(self):
        """重置 warm start 缓存（用于新 episode 开始）/ Reset the warm-start cache (used when a new episode begins)."""
        self._prev_control_points = None
        self._latest_latent_control_points = None
        self._warm_start_counter = 0
        self._prev_initial_turn = None  # 重置初始转弯缓存 / reset the initial-turn cache

    def generate_trajectories(self, depth_images: torch.Tensor, target_position: List[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """生成一批轨迹 / Generate a batch of trajectories."""
        # 注意：不在这里重置种子！
        # Note: do NOT reset the seed here!
        # 使用 __init__ 中设置的固定全局种子，实现跨帧确定性；批内多样性由 torch.randn 的内部状态自动保证
        # Use the fixed global seed set in __init__ for cross-frame determinism; intra-batch diversity is ensured automatically by torch.randn's internal state

        timing = {}
        total_start = time.time()

        # 创建调度器 / Create the scheduler
        scheduler_start = time.time()
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            prediction_type=self.model.scheduler.config.prediction_type
        )
        dpm_scheduler.set_timesteps(self.config.num_inference_steps, device=self.config.device)
        timing['scheduler_init'] = time.time() - scheduler_start

        # 准备目标位置 / Prepare the target position
        end_relative_pose = torch.tensor([target_position], device=self.config.device)
        end_relative_pose_batch = end_relative_pose.repeat(self.config.batch_size, 1)
        depth_images_batch = depth_images.repeat(self.config.batch_size, 1, 1, 1, 1)

        # 使用 AMP（自动混合精度）推理 / Run inference with AMP (automatic mixed precision)
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        with torch.no_grad(), torch.autocast(device_type='cuda' if self.config.device=='cuda' else 'cpu', enabled=(self.config.device=='cuda'), dtype=amp_dtype):
            batch_size = depth_images_batch.shape[0]

            # 条件编码 / Condition encoding
            condition_start = time.time()
            cond_end = end_relative_pose_batch.to(self.config.device)
            if self.normalizer is not None:
                end_np = end_relative_pose_batch.detach().cpu().numpy()
                end_norm_np = self.normalizer.normalize(end_np)
                cond_end = torch.from_numpy(end_norm_np).to(device=self.config.device, dtype=end_relative_pose_batch.dtype)

            # 准备初始转弯条件（取自上一帧 CP1 与 CP2 的 Y 均值）
            # Prepare the initial-turn condition (the mean Y of CP1 and CP2 from the previous frame)
            initial_turn = None
            has_initial_turn = None
            if hasattr(self.model.condition_encoder, 'use_initial_turn') and self.model.condition_encoder.use_initial_turn:
                if self._prev_initial_turn is not None:
                    # 有历史转弯信息 / Historical turn information is available
                    initial_turn = torch.full((batch_size,), self._prev_initial_turn,
                                            device=self.config.device, dtype=cond_end.dtype)
                    has_initial_turn = torch.ones(batch_size, dtype=torch.bool, device=self.config.device)
                else:
                    # 冷启动：使用 null token（在 encoder 内部处理）
                    # Cold start: use a null token (handled inside the encoder)
                    initial_turn = None
                    has_initial_turn = torch.zeros(batch_size, dtype=torch.bool, device=self.config.device)

            encoder_hidden_states = self.model.condition_encoder(
                depth_images_batch, cond_end,
                initial_turn=initial_turn,
                has_initial_turn=has_initial_turn
            ).to(dtype=self.model.unet.dtype)
            timing['condition_encoding'] = time.time() - condition_start

            # 初始化噪声 / Initialize noise
            init_start = time.time()

            # 纯随机初始化控制点 / Initialize control points purely at random
            randn_start = time.time()
            control_points, start_step_idx = self._initialize_control_points(batch_size, dpm_scheduler, self.model.unet.dtype)
            timing['warm_start_used'] = bool(self._warm_start_enabled and self._prev_control_points is not None)
            timing['randn_generation'] = time.time() - randn_start

            # 处理第一个控制点（固定为原点）/ Handle the first control point (fixed at the origin)
            fix_cp_start = time.time()
            if self.model.fix_first_cp_zero:
                if self._normalized_zero_cache is not None:
                    # 使用预计算的缓存值，避免重复的 numpy 计算和 CPU->GPU 传输
                    # Use the precomputed cached value to avoid repeated numpy computation and CPU->GPU transfers
                    control_points[:, :, 0] = self._normalized_zero_cache.view(1, 3)
                else:
                    control_points[:, :, 0] = 0.0
            timing['fix_first_cp'] = time.time() - fix_cp_start

            timing['noise_init'] = time.time() - init_start

            # 去噪过程 / Denoising process
            sampling_start = time.time()
            step_times = []  # 记录每步耗时 / record per-step timing

            timesteps_to_run = dpm_scheduler.timesteps[start_step_idx:]
            for step_idx, t in enumerate(timesteps_to_run):
                step_start = time.time()

                # 1. 准备 timestep tensor / Prepare the timestep tensor
                prep_start = time.time()
                timesteps = torch.full((batch_size,), t, device=self.config.device, dtype=torch.long)
                model_in = dpm_scheduler.scale_model_input(control_points, t)
                prep_time = time.time() - prep_start

                # 2. UNet 前向推理（主要耗时）/ UNet forward pass (the main cost)
                unet_start = time.time()
                v_pred = self.model.unet(
                    sample=model_in,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                unet_time = time.time() - unet_start

                # 3. 调度器单步更新 / Scheduler step
                scheduler_step_start = time.time()
                control_points = dpm_scheduler.step(v_pred, t, control_points).prev_sample
                scheduler_step_time = time.time() - scheduler_step_start

                # 4. 重置第一个控制点 / Reset the first control point
                fix_start = time.time()
                if self.model.fix_first_cp_zero:
                    if self._normalized_zero_cache is not None:
                        control_points[:, :, 0] = self._normalized_zero_cache.view(1, 3)
                    else:
                        control_points[:, :, 0] = 0.0
                fix_time = time.time() - fix_start

                step_total = time.time() - step_start
                step_times.append({
                    'step': step_idx,
                    'total': step_total * 1000,
                    'prep': prep_time * 1000,
                    'unet': unet_time * 1000,
                    'scheduler_step': scheduler_step_time * 1000,
                    'fix_cp': fix_time * 1000
                })

            timing['sampling'] = time.time() - sampling_start
            timing['step_details'] = step_times  # 保存每步详情 / save per-step details

            # 后处理 / Post-processing
            post_start = time.time()
            self._latest_latent_control_points = control_points.detach().cpu()
            control_points = control_points.transpose(1, 2)  # (batch_size, 8, 3)

            if self.normalizer is not None:
                # 关键：反归一化前先 clamp 到 [-1, 1]，防止外推放大，确保训练-推理一致性
                # Key: clamp to [-1, 1] before denormalization to prevent extrapolation blow-up and ensure training-inference consistency
                control_points = control_points.clamp(-1.0, 1.0)
                cp_np = control_points.detach().cpu().numpy()
                cp_denorm_np = self.normalizer.denormalize(cp_np)
                control_points = torch.from_numpy(cp_denorm_np).to(device=self.config.device, dtype=control_points.dtype)

            timing['post_processing'] = time.time() - post_start

        timing['total'] = time.time() - total_start

        # 打印详细耗时统计 / Print detailed timing statistics
        if self.config.show_verbose:
            print("\n⏱️  轨迹生成耗时详情:")
            print(f"   ├─ 调度器初始化: {timing.get('scheduler_init', 0)*1000:.2f}ms")
            print(f"   ├─ 条件编码:     {timing.get('condition_encoding', 0)*1000:.2f}ms")
            print(f"   ├─ 噪声初始化:   {timing.get('noise_init', 0)*1000:.2f}ms")
            print(f"   │  ├─ 随机数生成:   {timing.get('randn_generation', 0)*1000:.2f}ms")
            print(f"   │  └─ 修正第一点:   {timing.get('fix_first_cp', 0)*1000:.2f}ms")
            print(f"   ├─ DDPM采样:     {timing.get('sampling', 0)*1000:.2f}ms  ← 主要耗时")

            # 打印每步详情 / Print per-step details
            if 'step_details' in timing and timing['step_details']:
                print(f"   │  ┌─ DDPM采样步骤详情:")
                for step_info in timing['step_details']:
                    print(f"   │  ├─ Step {step_info['step']}: {step_info['total']:.2f}ms")
                    print(f"   │  │  ├─ 准备:         {step_info['prep']:.3f}ms")
                    print(f"   │  │  ├─ UNet推理:     {step_info['unet']:.2f}ms  ← 主要耗时")
                    print(f"   │  │  ├─ Scheduler步:  {step_info['scheduler_step']:.3f}ms")
                    print(f"   │  │  └─ 修正控制点:   {step_info['fix_cp']:.3f}ms")

                # 计算平均值 / Compute averages
                avg_unet = np.mean([s['unet'] for s in timing['step_details']])
                avg_total = np.mean([s['total'] for s in timing['step_details']])
                print(f"   │  └─ 平均每步: {avg_total:.2f}ms (UNet: {avg_unet:.2f}ms)")

            print(f"   ├─ 后处理:       {timing.get('post_processing', 0)*1000:.2f}ms")
            print(f"   └─ 总计:         {timing.get('total', 0)*1000:.2f}ms")
            print(f"   DDPM采样占比: {timing.get('sampling', 0)/max(timing.get('total', 0.001), 0.001)*100:.1f}%")

        return control_points.detach().cpu().numpy(), timing
