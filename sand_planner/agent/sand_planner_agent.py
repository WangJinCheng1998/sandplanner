#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner Agent（简化版本）/ SanD-planner Agent (simplified version).

直接使用推理引擎，只实现必要的 step_pointgoal 接口。
Directly uses the inference engine and implements only the essential
step_pointgoal interface.
"""

import os
import sys
import numpy as np
import tempfile
import time
import torch

# 设置 matplotlib 使用非交互式后端，避免 GUI 相关的错误
# Set matplotlib to a non-interactive backend to avoid GUI-related errors
import matplotlib
# 使用 Anti-Grain Geometry 后端，不需要 X11 或其他 GUI
# Use the Anti-Grain Geometry backend; no X11 or other GUI required
matplotlib.use('Agg')

# 条件导入 cv2，测试时可能不需要
# Conditionally import cv2; it may not be needed during testing
try:
    import cv2
except ImportError:
    print("Warning: cv2 not available, some visualization features may not work")
    cv2 = None

from sand_planner.config import InferenceConfig
from sand_planner.core.orchestrator import SandPlannerInference
from sand_planner.agent.depth_processor import ArrayDepthProcessor as DepthProcessor


class SandPlannerAgent:
    def __init__(self, image_intrinsic, config: InferenceConfig = None, verbose=False):

        self.image_intrinsic = image_intrinsic
        self.verbose = verbose
        self.config = config if config is not None else InferenceConfig()

        self.device = self.config.device
        self.predict_size = self.config.predict_size
        self.image_size = self.config.image_width
        self.memory_size = self.config.sequence_length
        self.temporal_depth = self.config.sequence_length
        self.default_behavior = self.config.default_behavior

        self.warmup_enabled = False
        self.warmup_strategy = False

        if hasattr(image_intrinsic, 'shape') and len(image_intrinsic.shape) == 2:
            self.config.camera_fx = float(image_intrinsic[0, 0])
            self.config.camera_fy = float(image_intrinsic[1, 1])
            self.config.camera_ppx = float(image_intrinsic[0, 2])
            self.config.camera_ppy = float(image_intrinsic[1, 2])

        self.planner = SandPlannerInference(self.config, verbose=self.verbose)
        self.depth_processor = DepthProcessor(self.config)
        self.temp_dir = tempfile.mkdtemp()

        if self.verbose:
            print(f"✅ SanD-planner Agent initialized:")
            print(f"   - Model: {self.config.checkpoint_path}")
            print(f"   - Device: {self.device}")
            print(f"   - Predict size: {self.predict_size}")

    def reset(self, batch_size, threshold):
        """重置 agent 状态 / Reset the agent state."""
        self.batch_size = batch_size
        self.stop_threshold = threshold

        # 初始化帧内存队列，每个 batch 环境一个队列
        # Initialize the frame memory queue, one queue per batch environment
        self.frame_memory_size = self.config.sequence_length  # 从配置读取 / read from config
        self.frame_memory_queue = [[] for i in range(batch_size)]

        # 初始化帧计数器，用于跳帧逻辑（每个环境独立计数）
        # Initialize frame counters for frame-skipping logic (per-environment count)
        self.frame_counters = [0 for i in range(batch_size)]

        # 初始化每个环境的深度特征缓存
        # Initialize the depth feature cache for each environment
        self.depth_caches = []
        for i in range(batch_size):
            cache = DepthProcessor(self.config)
            self.depth_caches.append(cache)

        # 初始化 mapper 步数计数器（用于定期清理 CUDA 内存）
        # Initialize the mapper step counter (used to periodically free CUDA memory)
        self.mapper_step_count = 0
        self.mapper_reset_interval = self.config.mapper_reset_interval  # 从配置读取 / read from config

        if self.verbose:
            print(f"🔄 Agent reset: batch_size={batch_size}, threshold={threshold}")
            print(f"📚 初始化{self.frame_memory_size}帧内存队列")

    def reset_env(self, i):
        """重置特定环境并清理 CUDA 内存防止泄漏 / Reset a specific environment and free CUDA memory to avoid leaks."""
        # 1. 清理对应环境的深度特征缓存
        # 1. Clear the depth feature cache of the corresponding environment
        if hasattr(self, 'depth_caches') and i < len(self.depth_caches):
            self.depth_caches[i].clear_cache()
            if self.verbose:
                print(f"🔄 重置环境{i}的深度特征缓存")

        # 1.5. 重置对应环境的帧计数器
        # 1.5. Reset the frame counter of the corresponding environment
        if hasattr(self, 'frame_counters') and i < len(self.frame_counters):
            self.frame_counters[i] = 0

        # 1.6. 定期重置 NVBlox mapper 防止内存累积
        # 1.6. Periodically reset the NVBlox mapper to prevent memory buildup
        if hasattr(self, 'mapper_step_count'):
            self.mapper_step_count += 1
            if self.mapper_step_count >= self.mapper_reset_interval:
                if self.verbose:
                    print(f"🧹 达到{self.mapper_reset_interval}步，重置NVBlox mapper防止内存泄漏...")
                self.planner.reset_environment()
                self.mapper_step_count = 0  # 重置计数器 / reset the counter
            if self.verbose:
                print(f"🔄 重置环境{i}的帧计数器")

        # 2. 清理 NVBlox mapper 和温启动缓存（关键，防止 CUDA 内存泄漏）
        # 2. Clear the NVBlox mapper and warmup cache (critical to prevent CUDA memory leaks)
        if hasattr(self, 'planner') and self.planner is not None:
            try:
                self.planner.reset_environment()
                if self.verbose:
                    print(f"✅ 环境{i}: 已清理NVBlox mapper和温启动缓存")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 环境{i}: 清理失败 - {e}")

        # 3. 额外的 CUDA 内存清理（保险起见）
        # 3. Additional CUDA memory cleanup (just to be safe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"📊 环境{i}重置后: GPU内存 {mem_allocated:.2f}GB / {mem_reserved:.2f}GB")

    def update_camera_config(self, image_intrinsic):
        """更新相机内参配置 / Update the camera intrinsics configuration."""
        if hasattr(image_intrinsic, 'shape') and len(image_intrinsic.shape) == 2:
            self.config.camera_fx = float(image_intrinsic[0, 0])
            self.config.camera_fy = float(image_intrinsic[1, 1])
            self.config.camera_ppx = float(image_intrinsic[0, 2])
            self.config.camera_ppy = float(image_intrinsic[1, 2])
            if self.verbose:
                print(f"📷 相机内参已更新: fx={self.config.camera_fx}, fy={self.config.camera_fy}")
                print(f"                  ppx={self.config.camera_ppx}, ppy={self.config.camera_ppy}")



    def _warmup_model(self):
        """
        策略化模型预热：根据策略配置执行预热 / Strategy-based model warmup driven by the strategy configuration.

        彻底解决第一次调用时 "No trajectory available" 的问题。
        Fully resolves the "No trajectory available" issue on the first call.
        """
        try:
            # 导入预热配置
            # Import the warmup configuration
            from warmup_config import WarmupConfig, choose_warmup_strategy

            # 获取预热配置
            # Obtain the warmup configuration
            if isinstance(self.warmup_strategy, str):
                if self.warmup_strategy in ["development", "testing", "production", "server", "demo"]:
                    warmup_config = choose_warmup_strategy(self.warmup_strategy)
                else:
                    warmup_config = WarmupConfig(self.warmup_strategy)
            else:
                warmup_config = WarmupConfig("standard")  # 默认策略 / default strategy

            if self.verbose:
                print("🔥 开始策略化模型预热...")
                print(f"   策略: {warmup_config.strategy_name}")
                print(f"   描述: {warmup_config.description}")
                print(f"   预热轮数: {warmup_config.warmup_rounds}")
                print(f"   预估时间: {warmup_config.get_total_estimated_time():.1f}秒")

            total_start_time = time.time()

            # 创建多样化的虚拟输入数据
            # Create diverse dummy input data
            dummy_templates = [
                {"rgb_range": (0, 255), "depth_scale": 3.0, "goal": [2.0, 0.0, 0.5]},
                {"rgb_range": (50, 200), "depth_scale": 8.0, "goal": [1.5, -1.0, 0.3]},
                {"rgb_range": (100, 255), "depth_scale": 5.0, "goal": [3.0, 1.5, 0.8]},
                {"rgb_range": (20, 180), "depth_scale": 6.0, "goal": [2.5, -0.5, 0.6]},
                {"rgb_range": (80, 240), "depth_scale": 4.0, "goal": [1.8, 1.0, 0.4]}
            ]

            successful_warmups = 0

            # 执行指定轮数的预热
            # Run the specified number of warmup rounds
            for round_idx in range(warmup_config.warmup_rounds):
                try:
                    # 循环使用模板
                    # Cycle through the templates
                    template = dummy_templates[round_idx % len(dummy_templates)]

                    # 生成虚拟数据
                    # Generate dummy data
                    dummy_rgb = np.random.randint(
                        template["rgb_range"][0],
                        template["rgb_range"][1],
                        (1, 480, 640, 3),
                        dtype=np.uint8
                    )
                    dummy_depth = np.random.rand(1, 480, 640, 1).astype(np.float32) * template["depth_scale"]
                    dummy_goal = np.array([template["goal"]])

                    if self.verbose:
                        print(f"   轮次 {round_idx + 1}/{warmup_config.warmup_rounds}: ", end="")

                    warmup_start = time.time()
                    trajectory, _, _, _ = self.act(
                        images=dummy_rgb,
                        depth_images=dummy_depth,
                        goals=dummy_goal
                    )
                    warmup_time = time.time() - warmup_start

                    # 检查是否返回有效轨迹
                    # Check whether a valid trajectory was returned
                    if trajectory is not None and np.any(trajectory != 0):
                        successful_warmups += 1
                        if self.verbose:
                            print(f"✅ 成功 ({warmup_time:.3f}秒)")
                    else:
                        if self.verbose:
                            print(f"❌ 空轨迹 ({warmup_time:.3f}秒)")

                    # 轮间等待（除了最后一轮）
                    # Wait between rounds (except after the last round)
                    if round_idx < warmup_config.warmup_rounds - 1:
                        time.sleep(warmup_config.wait_between_rounds)

                except Exception as e:
                    if self.verbose:
                        print(f"❌ 异常 - {e}")

            # 最终等待
            # Final wait
            if self.verbose:
                print(f"   最终等待: {warmup_config.final_wait}秒...")

            time.sleep(warmup_config.final_wait)

            total_time = time.time() - total_start_time
            success_rate = successful_warmups / warmup_config.warmup_rounds * 100

            if self.verbose:
                print(f"✅ 策略化预热完成！")
                print(f"   成功率: {successful_warmups}/{warmup_config.warmup_rounds} ({success_rate:.1f}%)")
                print(f"   实际耗时: {total_time:.3f}秒")

                if success_rate >= 80:
                    print("   🎉 预热效果优秀，系统就绪")
                elif success_rate >= 50:
                    print("   ✅ 预热效果良好，可正常使用")
                else:
                    print("   ⚠️ 预热效果一般，首次推理可能较慢")

        except ImportError:
            if self.verbose:
                print("⚠️ 预热配置模块不可用，使用简化预热")
            self._simple_warmup()
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 预热过程异常: {e}")
                print("   使用简化预热...")
            self._simple_warmup()

    def _simple_warmup(self):
        """简化预热方案（当配置化预热失败时使用）/ Simplified warmup fallback used when configured warmup fails."""
        try:
            if self.verbose:
                print("🔥 执行简化预热...")

            dummy_rgb = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
            dummy_depth = np.random.rand(1, 480, 640, 1).astype(np.float32) * 5.0
            dummy_goal = np.array([[2.0, 0.0, 0.5]])

            start_time = time.time()
            self.act(images=dummy_rgb, depth_images=dummy_depth, goals=dummy_goal)
            warmup_time = time.time() - start_time

            time.sleep(2.0)  # 固定等待时间 / fixed wait time

            if self.verbose:
                print(f"   简化预热完成: {warmup_time:.3f}秒")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 简化预热也失败: {e}")
                print("   进行最终等待...")
            time.sleep(3.0)

    def process_pointgoal(self, goals):
        """
        处理点目标坐标，应用定制化的裁剪逻辑 / Process point-goal coordinates with customized clipping logic.

        Args:
            goals: (batch_size, 3) 目标点坐标 / point-goal coordinates.

        Returns:
            clip_goals: 裁剪后的目标点坐标 / clipped point-goal coordinates.
        """
        clip_goals = goals.copy()
        # x 坐标裁剪到前方距离范围 / clip the x coordinate (forward distance) to range. 备选/alt: 0, 10
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], -3, 10)
        # y 坐标裁剪到左右距离范围 / clip the y coordinate (lateral distance) to range. 备选/alt: -10, 3
        clip_goals[:, 1] = np.clip(clip_goals[:, 1], -10, 10)
        return clip_goals

    def _generate_default_trajectory(self, batch_size=1, num_trajectories=16):
        """根据 default_behavior 参数生成默认轨迹 / Generate a default trajectory based on the default_behavior parameter."""
        if self.default_behavior == "stop":
            # 停住不动：所有轨迹点都是 [0, 0, 0]
            # Stop in place: all trajectory points are [0, 0, 0]
            best_trajectory = np.zeros((batch_size, self.predict_size, 3))
            all_trajectories = np.zeros((batch_size, num_trajectories, self.predict_size, 3))
            values = np.full((batch_size, num_trajectories), float('inf'))

            if self.verbose:
                print("🛑 使用默认行为: 停住不动")

        elif self.default_behavior == "forward":
            # 缓慢前进：生成轻微的前进轨迹
            # Move forward slowly: generate a slight forward trajectory
            best_trajectory = np.zeros((batch_size, self.predict_size, 3))
            all_trajectories = np.zeros((batch_size, num_trajectories, self.predict_size, 3))

            # 生成缓慢前进轨迹（每步前进 5cm）
            # Generate a slow forward trajectory (5cm forward per step)
            for i in range(self.predict_size):
                x_step = (i + 1) * 0.05  # 每步前进 5cm / 5cm forward per step
                best_trajectory[:, i, 0] = x_step
                all_trajectories[:, :, i, 0] = x_step

            values = np.full((batch_size, num_trajectories), 10.0)  # 给予中等代价 / assign a medium cost

            if self.verbose:
                print("➡️ 使用默认行为: 缓慢前进")
        else:
            # 默认还是停住
            # Default to stopping in place
            best_trajectory = np.zeros((batch_size, self.predict_size, 3))
            all_trajectories = np.zeros((batch_size, num_trajectories, self.predict_size, 3))
            values = np.full((batch_size, num_trajectories), float('inf'))

            if self.verbose:
                print("🛑 未知默认行为，使用停住不动")

        return best_trajectory, all_trajectories, values

    def _save_depth_as_temp_file(self, depth_array):
        """将深度数组保存为临时 PNG 文件（保留用于向后兼容）/ Save the depth array as a temporary PNG file (kept for backward compatibility)."""
        # 转换深度数组为毫米单位的 uint16
        # Convert the depth array to uint16 in millimeters
        depth_mm = (depth_array * 1000).astype(np.uint16)

        # 如果是 4D 数组，取第一个 batch 和通道
        # If it is a 4D array, take the first batch and channel
        if len(depth_mm.shape) == 4:
            depth_mm = depth_mm[0, :, :, 0]
        elif len(depth_mm.shape) == 3:
            depth_mm = depth_mm[:, :, 0]

        # 保存为临时文件
        # Save as a temporary file
        temp_file = os.path.join(self.temp_dir, "temp_depth.png")
        if cv2 is not None:
            cv2.imwrite(temp_file, depth_mm)
        else:
            # 如果 cv2 不可用，使用 PIL 或 numpy 保存
            # If cv2 is unavailable, save with PIL or numpy
            from PIL import Image
            Image.fromarray(depth_mm).save(temp_file)

        return temp_file

    def _process_depth_for_inference(self, depth_arrays: np.ndarray) -> torch.Tensor:
        """直接处理深度数组用于推理，避免临时文件 IO / Process depth arrays directly for inference, avoiding temporary-file IO.

        Args:
            depth_arrays: (batch_size, H, W, 1) 深度数组 / depth arrays.

        Returns:
            torch.Tensor: (batch_size, seq_len, 1, H, W) 处理后的深度序列 / processed depth sequences.
        """
        if self.verbose:
            print(f"🔄 直接处理深度数组: {depth_arrays.shape}")

        # 使用深度处理器编码
        # Encode using the depth processor
        depth_sequences = self.depth_processor.encode_depth_for_model(depth_arrays)

        if self.verbose:
            print(f"✅ 深度编码完成: {depth_sequences.shape}")

        return depth_sequences

    def _process_depth_with_cache(self, depth_arrays: np.ndarray) -> torch.Tensor:
        """使用特征缓存的高效深度处理 / Efficient depth processing using the feature cache.

        Args:
            depth_arrays: (batch_size, H, W, 1) 深度数组 / depth arrays.

        Returns:
            torch.Tensor: (batch_size, seq_len, 1, H, W) 处理后的深度序列 / processed depth sequences.
        """
        batch_size = depth_arrays.shape[0]
        cached_sequences = []

        # 保存原始深度用于 ESDF（避免精度损失）
        # Keep the original depth for ESDF (to avoid precision loss)
        if not hasattr(self, 'original_depth_for_esdf'):
            self.original_depth_for_esdf = None

        # 保存第一个环境的最新深度帧用于 ESDF
        # Keep the latest depth frame of the first environment for ESDF
        if batch_size > 0:
            if len(depth_arrays.shape) == 4:  # (batch_size, H, W, 1)
                # 取第一个环境，移除通道维度
                # Take the first environment and drop the channel dimension
                original_depth = depth_arrays[0, :, :, 0]
            else:  # (batch_size, H, W)
                original_depth = depth_arrays[0]

            # 对原始深度进行基本预处理（单位转换 + 裁剪，但保持原分辨率）
            # Apply basic preprocessing to the original depth (unit conversion + clipping, keeping the original resolution)
            if original_depth.dtype != np.float32:
                original_depth = original_depth.astype(np.float32)

            # 单位转换
            # Unit conversion
            if original_depth.max() > 50:  # 假设大于 50 的是毫米单位 / assume values above 50 are in millimeters
                original_depth = original_depth / 1000.0

            # 裁剪深度值（保持原分辨率）
            # Clip depth values (keeping the original resolution)
            original_depth = np.clip(original_depth, 0, self.config.max_depth)

            # 归一化到 [0, 1]（为了与后续 ESDF 处理保持一致）
            # Normalize to [0, 1] (to stay consistent with the later ESDF processing)
            self.original_depth_for_esdf = original_depth / self.config.max_depth

        for i in range(batch_size):
            # 获取单帧深度
            # Obtain a single depth frame
            if len(depth_arrays.shape) == 4:  # (batch_size, H, W, 1)
                single_depth = depth_arrays[i, :, :, 0]  # 移除最后一个维度 / drop the last dimension
            else:  # (batch_size, H, W)
                single_depth = depth_arrays[i]

            # 添加到对应环境的缓存（每帧都保存）
            # Add it to the corresponding environment's cache (save every frame)
            self.depth_caches[i].add_frame_to_cache(single_depth, should_save=True)

            # 增加帧计数器
            # Increment the frame counter
            self.frame_counters[i] += 1

            # 从缓存构建序列
            # Build the sequence from the cache
            sequence = self.depth_caches[i].get_sequence_from_cache()  # (1, seq_len, 1, H, W)
            cached_sequences.append(sequence[0])  # 去掉 batch 维度 / drop the batch dimension

            if self.verbose:
                cache_info = self.depth_caches[i].get_cache_info()
                print(f"📚 环境{i}缓存状态: {cache_info['cache_size']}/{cache_info['max_cache_size']} 帧, 已满={cache_info['is_full']}, 当前帧{self.frame_counters[i]}")

        # 堆叠成批次 (batch_size, seq_len, 1, H, W)
        # Stack into a batch (batch_size, seq_len, 1, H, W)
        result = torch.stack(cached_sequences, dim=0)

        if self.verbose:
            print(f"✅ 缓存深度编码完成: {result.shape}")

        return result

    def _fallback_to_temp_file_inference(self, current_depths, goals, images):
        """后备方案：使用临时文件方式进行推理 / Fallback: run inference via temporary files."""
        if self.verbose:
            print("🔄 使用临时文件后备方案...")

        try:
            # 保存为临时文件
            # Save to a temporary file
            temp_depth_file = self._save_depth_as_temp_file(current_depths)
            depth_filename = os.path.basename(temp_depth_file)

            # 临时更新配置的输入目录
            # Temporarily update the configured input directory
            original_input_dir = self.config.input_depth_dir
            self.config.input_depth_dir = self.temp_dir

            try:
                # 调用原始推理方法
                # Call the original inference method
                results = self.planner.run_inference(depth_filename)
                return self._process_inference_results(results, goals, images)
            finally:
                # 恢复原始配置
                # Restore the original configuration
                self.config.input_depth_dir = original_input_dir
                # 清理临时文件
                # Clean up the temporary file
                if os.path.exists(temp_depth_file):
                    os.remove(temp_depth_file)
        except Exception as e:
            if self.verbose:
                print(f"❌ 后备方案也失败: {e}")
            # 返回默认轨迹
            # Return a default trajectory
            best_trajectory, all_trajectories, values = self._generate_default_trajectory(batch_size=1, num_trajectories=16)
            trajectory_mask = images.copy() if images is not None else np.zeros((1, 224, 224, 3))
            return best_trajectory, all_trajectories, values, trajectory_mask

    def step_pointgoal(self, goals, images, depths):
        """
        点目标导航主接口 / Point-goal navigation, the main interface.

        Args:
            goals: (batch_size, 3) 目标点坐标 / point-goal coordinates.
            images: (batch_size, H, W, 3) RGB 图像（保持接口兼容，但不使用）/ RGB images (kept for interface compatibility but unused).
            depths: (batch_size, H, W, 1) 深度图像（用于多帧内存和推理）/ depth images (used for multi-frame memory and inference).

        Returns:
            best_trajectory: (predict_size, 3) 最佳轨迹（完整轨迹点序列）/ best trajectory (full sequence of trajectory points).
            all_trajectories: (num_trajectories, predict_size, 3) 所有轨迹，按 cost 从小到大排序，
                第一个轨迹是 cost 最小（最佳）的轨迹 /
                all trajectories, sorted by cost ascending; the first one is the lowest-cost (best) trajectory.
            values: (num_trajectories,) 轨迹代价评分，按从小到大排序 / trajectory cost scores, sorted ascending.
                包含清距惩罚、轨迹长度、目标误差的加权总和，代价越小表示轨迹越好 /
                A weighted sum of the clearance penalty, trajectory length, and goal error; lower cost means a better trajectory:
                    - 清距惩罚：基于 ESDF 的安全距离违规惩罚（权重 1000.0）/ clearance penalty: ESDF-based safety-margin violation penalty (weight 1000.0).
                    - 轨迹长度：路径效率惩罚（权重 1.0）/ trajectory length: path-efficiency penalty (weight 1.0).
                    - 目标误差：终点与目标点的距离误差（权重 1.0）/ goal error: distance error between the endpoint and the goal (weight 1.0).
                注意：轨迹和 cost 都已排序，all_trajectories[i] 对应 values[i] /
                Note: both trajectories and costs are sorted, with all_trajectories[i] corresponding to values[i].
            mask: 可视化掩码 / visualization mask.
        """

        # 0. 直接使用连续多帧深度序列进行推理
        # 0. Run inference directly on the consecutive multi-frame depth sequence
        if self.verbose:
            print(f"📦 直接处理深度输入: {depths.shape}")

        # 1. 处理并裁剪目标位置
        # 1. Process and clip the goal position
        if len(goals) > 0:
            processed_goals = self.process_pointgoal(goals)  # 应用点目标裁剪逻辑 / apply the point-goal clipping logic
            target_goal = processed_goals[0].tolist()  # 取第一个裁剪后的目标 / take the first clipped goal
            self.config.target_position = target_goal
            if self.verbose:
                print(f"🎯 设置目标 (原始): {goals[0].tolist()}")
                print(f"🎯 设置目标 (裁减): {target_goal}")

        # 2. 使用缓存的高效深度处理
        # 2. Use cached, efficient depth processing
        try:
            # 使用特征缓存的高效深度处理（直接使用输入的 depths）
            # Efficient depth processing using the feature cache (directly using the input depths)
            depth_sequences = self._process_depth_with_cache(depths)  # (batch_size, seq_len, 1, H, W)

            if self.verbose:
                print(f"📐 高效深度序列已准备: {depth_sequences.shape}")

            # 2.5. 将原始深度传递给推理引擎用于 ESDF
            # 注意：需要在调用 process_depth_arrays 之前设置，因为该方法内部会使用它
            # 2.5. Pass the original depth to the inference engine for ESDF
            # Note: it must be set before calling process_depth_arrays, since that method uses it internally
            if hasattr(self, 'original_depth_for_esdf') and self.original_depth_for_esdf is not None:
                # 直接在推理对象上设置临时属性
                # Set a temporary attribute directly on the inference object
                self.planner.agent_original_depth = self.original_depth_for_esdf
                if self.verbose:
                    print(f"📊 已传递原始深度用于ESDF: {self.original_depth_for_esdf.shape}")

            # 3. 调用新的直接深度数组处理方法
            # 3. Call the new direct depth-array processing method
            results = self.planner.process_depth_arrays(depth_sequences)

            # 4. 处理推理结果
            # 4. Process the inference results
            return self._process_inference_results(results, goals, images)
        except Exception as direct_inference_error:
            if self.verbose:
                print(f"❌ 直接深度处理失败: {direct_inference_error}")
                print("🔄 回退到临时文件方式...")
            # 取最新帧作为后备方案的输入
            # Use the latest frames as input for the fallback path
            current_depths = depths[:, :, :, :]  # 直接使用当前输入 / use the current input directly
            return self._fallback_to_temp_file_inference(current_depths, goals, images)

    def _process_inference_results(self, results, goals, images):
        """处理推理结果的公共方法 / Shared helper for processing inference results."""
        try:
            # 4. 提取结果并进行安全检查
            # 4. Extract results and run safety checks
            control_points = results.get('control_points')
            best_index = results.get('best_index', 0)
            sampled_trajectories = results.get('sampled_trajectories')

            # 安全检查
            # Safety check
            if control_points is None:
                raise ValueError("未获得控制点结果")

            # 确保 best_index 是有效的
            # Make sure best_index is valid
            if isinstance(best_index, (list, tuple, np.ndarray)):
                best_index = int(best_index[0]) if len(best_index) > 0 else 0
            else:
                best_index = int(best_index) if best_index is not None else 0

            # 确保 best_index 在有效范围内
            # Make sure best_index is within the valid range
            if len(control_points) > 0:
                best_index = max(0, min(best_index, len(control_points) - 1))
            else:
                best_index = 0

            if self.verbose:
                print(f"📊 SanD-planner推理完成:")
                print(f"   - 控制点形状: {control_points.shape}")
                print(f"   - 最佳轨迹索引: {best_index}")
                print(f"   - 采样轨迹: {sampled_trajectories is not None}")
                if sampled_trajectories is not None:
                    print(f"   - 采样轨迹数量: {len(sampled_trajectories)}")

            # 6. 获取排序后的轨迹代价评分和轨迹顺序
            # 6. Obtain the sorted trajectory cost scores and trajectory order
            values = None
            sorted_trajectory_indices = None

            if 'evaluation_results' in results and 'results' in results['evaluation_results']:
                # 提取排序后的评估结果（已按 total_cost 从小到大排序）
                # Extract the sorted evaluation results (already sorted by total_cost ascending)
                traj_results = results['evaluation_results']['results']
                values = np.array([r['total_cost'] for r in traj_results])
                sorted_trajectory_indices = [r['trajectory_id'] for r in traj_results]

                if self.verbose:
                    print(f"   - 轨迹代价评分: {values.shape}, 范围: [{values.min():.3f}, {values.max():.3f}]")
                    print(f"   - 排序后轨迹索引: {sorted_trajectory_indices[:5]}... (显示前5个)")
                    print(f"   - 最佳轨迹索引: {sorted_trajectory_indices[0]}, 代价: {values[0]:.3f}")

                    # 验证 values 数组是否已正确排序
                    # Verify that the values array is correctly sorted
                    is_sorted = all(values[i] <= values[i+1] for i in range(len(values)-1))
                    print(f"   - values数组已排序（从小到大）: {'✅' if is_sorted else '❌ 错误！'}")

                    # 打印前 3 条最佳轨迹的详细 cost 分解
                    # Print the detailed cost breakdown of the top 3 trajectories
                    print(f"\n📊 前3条最佳轨迹对比:")
                    for i in range(min(3, len(traj_results))):
                        r = traj_results[i]
                        print(f"   #{i+1} - ID:{r['trajectory_id']:2d}, "
                              f"Total:{r['total_cost']:8.3f}, "
                              f"Clear:{r['weighted_costs']['clearance']:8.3f}, "
                              f"Len:{r['weighted_costs']['length']:6.2f}, "
                              f"Goal:{r['weighted_costs']['goal']:6.2f}")
                        # 显示原始 cost（未加权）
                        # Show the raw (unweighted) cost
                        print(f"        (原始: clear={r['clearance_cost']:.4f}, "
                              f"len={r['length_cost']:.2f}, "
                              f"goal={r['goal_cost']:.2f})")

            # 7. 转换为标准输出格式并按 cost 排序
            # 7. Convert to the standard output format and sort by cost
            if sampled_trajectories is not None and len(sampled_trajectories) > 0:
                # 处理不均匀的采样轨迹：找到最大长度并填充
                # Handle unevenly sized sampled trajectories: find the maximum length and pad
                max_length = max(len(traj) for traj in sampled_trajectories)
                if self.verbose:
                    print(f"   - 采样轨迹长度范围: {min(len(traj) for traj in sampled_trajectories)} - {max_length}")

                # 填充到统一长度
                # Pad to a uniform length
                padded_trajectories = []
                for traj in sampled_trajectories:
                    if len(traj) < max_length:
                        # 用最后一个点填充
                        # Pad with the last point
                        padding = np.tile(traj[-1:], (max_length - len(traj), 1))
                        padded_traj = np.vstack([traj, padding])
                    else:
                        padded_traj = traj
                    padded_trajectories.append(padded_traj)

                # 按排序顺序重新排列轨迹
                # Reorder trajectories into the sorted order
                if sorted_trajectory_indices is not None and values is not None:
                    # 按 cost 从小到大的顺序重排轨迹
                    # Reorder trajectories by cost ascending
                    sorted_trajectories = []
                    for idx in sorted_trajectory_indices:
                        # 安全检查索引范围
                        # Safety-check the index range
                        if 0 <= idx < len(padded_trajectories):
                            sorted_trajectories.append(padded_trajectories[idx])
                        else:
                            if self.verbose:
                                print(f"⚠️ 警告: 轨迹索引 {idx} 超出范围，使用第一个轨迹")
                            sorted_trajectories.append(padded_trajectories[0])

                    if len(sorted_trajectories) > 0:
                        # (num_trajectories, max_length, 3)，已排序 / sorted
                        all_trajectories = np.array(sorted_trajectories)
                        # 最佳轨迹是排序后的第一个
                        # The best trajectory is the first one after sorting
                        best_trajectory = all_trajectories[0]  # (max_length, 3)，完整轨迹 / full trajectory
                        # 验证轨迹重排的正确性
                        # Verify the correctness of the trajectory reordering
                        if self.verbose:
                            print(f"\n🔍 轨迹重排验证:")
                            print(f"   - 原始轨迹数: {len(padded_trajectories)}")
                            print(f"   - 重排后轨迹数: {len(sorted_trajectories)}")
                            print(f"   - 最佳轨迹索引（在原始列表中）: {sorted_trajectory_indices[0]}")
                            print(f"   - 最佳轨迹cost: {values[0]:.3f}")
                            print(f"   - sorted_trajectories[0] == padded_trajectories[{sorted_trajectory_indices[0]}]: "
                                  f"{'✅' if np.array_equal(sorted_trajectories[0], padded_trajectories[sorted_trajectory_indices[0]]) else '❌'}")
                    else:
                        # 如果没有有效轨迹，使用第一个填充轨迹
                        # If there are no valid trajectories, use the first padded trajectory
                        all_trajectories = np.array(padded_trajectories)
                        if len(padded_trajectories) > 0:
                            best_trajectory = padded_trajectories[0]
                        else:
                            # 使用默认行为生成轨迹
                            # Generate a trajectory using the default behavior
                            default_best, _, _ = self._generate_default_trajectory(batch_size=1, num_trajectories=1)
                            best_trajectory = default_best[0]  # 去掉 batch 维度: (predict_size, 3) / drop the batch dimension: (predict_size, 3)

                    if self.verbose:
                        print(f"   - 轨迹已按cost排序: {all_trajectories.shape}")
                        print(f"   - 最佳轨迹cost: {values[0]:.3f}")
                else:
                    # 如果没有排序信息，使用原始顺序
                    # If there is no sorting information, keep the original order
                    if len(padded_trajectories) > 0:
                        all_trajectories = np.array(padded_trajectories)  # (num_trajectories, max_length, 3)
                        # 安全检查 best_index
                        # Safety-check best_index
                        safe_best_index = max(0, min(best_index, len(padded_trajectories) - 1))
                        best_trajectory = all_trajectories[safe_best_index]  # (max_length, 3)，完整轨迹 / full trajectory

                        # 如果没有 cost 信息，创建默认值
                        # If there is no cost information, create default values
                        if values is None:
                            values = np.zeros(len(sampled_trajectories))
                            if self.verbose:
                                print(f"   - 使用默认零代价: {values.shape}")

                        if self.verbose:
                            print(f"   - 使用原始轨迹顺序: {all_trajectories.shape}")
                    else:
                        # 没有有效的采样轨迹
                        # No valid sampled trajectories
                        if self.verbose:
                            print("⚠️ 警告: 没有有效的采样轨迹，使用默认轨迹")
                        best_trajectory, all_trajectories, values = self._generate_default_trajectory(batch_size=1, num_trajectories=1)
                        all_trajectories = all_trajectories  # 保持 (1, 1, predict_size, 3) 格式 / keep the (1, 1, predict_size, 3) format
                        values = values  # 保持 (1, 1) 格式 / keep the (1, 1) format

            else:
                # 回退到控制点（如果采样失败）
                # Fall back to control points (if sampling failed)
                print("   - 警告: 未找到采样轨迹，使用控制点")

                # 如果有排序信息，按排序顺序重排控制点
                # If sorting information exists, reorder control points into the sorted order
                if sorted_trajectory_indices is not None and values is not None and len(control_points) > 0:
                    sorted_control_points = []
                    for idx in sorted_trajectory_indices:
                        if 0 <= idx < len(control_points):
                            sorted_control_points.append(control_points[idx])
                        else:
                            if self.verbose:
                                print(f"⚠️ 警告: 控制点索引 {idx} 超出范围，使用第一个控制点")
                            if len(control_points) > 0:
                                sorted_control_points.append(control_points[0])

                    if len(sorted_control_points) > 0:
                        all_trajectories = np.array(sorted_control_points)  # 已排序的控制点 / sorted control points
                        # 最佳（最低 cost），完整控制点序列
                        # Best (lowest cost): the full control-point sequence
                        best_trajectory = all_trajectories[0]
                        if self.verbose:
                            print(f"   - 控制点已按cost排序: {all_trajectories.shape}")
                    else:
                        # 没有有效的排序控制点
                        # No valid sorted control points
                        all_trajectories = control_points
                        safe_best_index = max(0, min(best_index, len(control_points) - 1))
                        best_trajectory = control_points[safe_best_index]
                        if self.verbose:
                            print(f"   - 控制点使用原始顺序: {all_trajectories.shape}")
                else:
                    # 没有排序信息，使用原始顺序
                    # No sorting information; keep the original order
                    if len(control_points) > 0:
                        all_trajectories = control_points  # (batch_size, 8, 3)
                        safe_best_index = max(0, min(best_index, len(control_points) - 1))
                        best_trajectory = control_points[safe_best_index]  # (8, 3)，完整控制点序列 / the full control-point sequence

                        # 为控制点设置默认代价值，如果之前没有设置的话
                        # Set default cost values for the control points if they were not set before
                        if values is None:
                            values = np.zeros(len(control_points))
                            values[safe_best_index] = -1.0  # 给最佳轨迹一个较好的分数 / give the best trajectory a better score
                            if self.verbose:
                                print(f"   - 控制点默认代价: {values.shape}")
                    else:
                        # 没有有效控制点
                        # No valid control points
                        if self.verbose:
                            print("❌ 错误: 没有有效的控制点")
                        best_trajectory, all_trajectories, values = self._generate_default_trajectory(batch_size=1, num_trajectories=1)
                        values = np.array([[float('inf')]])  # 输出格式: (batch_size, num_trajectories) / output format: (batch_size, num_trajectories)

            # trajectory_mask: 将轨迹投影到图像
            # 需要将轨迹格式转换为 (batch_size, num_trajectories, length, 3)
            # trajectory_mask: project trajectories onto the image
            # The trajectory format must be converted to (batch_size, num_trajectories, length, 3)
            trajectories_for_viz = all_trajectories[np.newaxis, :]  # 添加 batch 维度 / add the batch dimension
            values_for_viz = values[np.newaxis, :] if values is not None else None  # 添加 batch 维度 / add the batch dimension
            trajectory_mask = self.project_trajectory(images, trajectories_for_viz, values_for_viz)

            # 最终安全检查
            # Final safety check
            if best_trajectory is None or all_trajectories is None:
                if self.verbose:
                    print("❌ 错误: 轨迹为 None，使用默认轨迹")
                best_trajectory, all_trajectories, values = self._generate_default_trajectory(batch_size=1, num_trajectories=16)

            # 返回格式: execute_trajectory (1, 24, 3), all_trajectory (1, 16, 24, 3), all_values (1, 16)
            # Output format: execute_trajectory (1, 24, 3), all_trajectory (1, 16, 24, 3), all_values (1, 16)

            # 1. execute_trajectory: 保持 (1, trajectory_length, 3) 格式
            # 1. execute_trajectory: keep the (1, trajectory_length, 3) format
            if len(best_trajectory.shape) == 2:  # (trajectory_length, 3)
                best_trajectory_final = best_trajectory[np.newaxis, :]  # (1, trajectory_length, 3)
            else:
                best_trajectory_final = best_trajectory

            # 2. all_trajectory: 需要添加 batch 维度变为 (1, num_trajectories, trajectory_length, 3)
            # 2. all_trajectory: add a batch dimension to make it (1, num_trajectories, trajectory_length, 3)
            if len(all_trajectories.shape) == 3:  # (num_trajectories, trajectory_length, 3)
                all_trajectory_final = all_trajectories[np.newaxis, :]  # (1, num_trajectories, trajectory_length, 3)
            else:
                all_trajectory_final = all_trajectories

            # 3. all_values: 需要添加 batch 维度变为 (1, num_trajectories)
            # 3. all_values: add a batch dimension to make it (1, num_trajectories)
            if values is not None and len(values.shape) == 1:  # (num_trajectories,)
                values_final = values[np.newaxis, :]  # (1, num_trajectories)
            else:
                values_final = values

            # 8. ESDF 安全检查：检查最佳轨迹前 10 个点的 ESDF 最小值
            # 8. ESDF safety check: inspect the minimum ESDF over the first 10 points of the best trajectory
            esdf_safety_threshold = self.config.esdf_safety_threshold  # 从配置读取 / read from config
            need_rotation = False

            if ('evaluation_results' in results and 'results' in results['evaluation_results'] and
                len(results['evaluation_results']['results']) > 0):

                # 获取最佳轨迹的评估详情
                # Get the evaluation details of the best trajectory
                best_result = results['evaluation_results']['results'][0]  # 已排序，第一个是最佳的 / sorted, the first one is the best

                if ('details' in best_result and 'clearance' in best_result['details'] and
                    'distances' in best_result['details']['clearance']):

                    esdf_distances = best_result['details']['clearance']['distances']

                    # 检查前若干个点（若轨迹点数不足，则检查全部点）
                    # Check the leading points (if the trajectory has fewer points, check all of them)
                    check_points = min(10, len(esdf_distances))
                    front_distances = esdf_distances[:check_points]
                    min_esdf = np.min(front_distances)

                    if self.verbose:
                        print(f"🔍 ESDF安全检查:")
                        print(f"   - 检查前 {check_points} 个点")
                        print(f"   - ESDF最小距离: {min_esdf:.3f}m")
                        print(f"   - 安全阈值: {esdf_safety_threshold}m")

                    # 如果最小距离小于阈值，需要旋转避障
                    # If the minimum distance is below the threshold, rotate to avoid the obstacle
                    if min_esdf < esdf_safety_threshold:
                        need_rotation = True
                        if self.verbose:
                            print(f"⚠️ 检测到前方障碍物过近 (min_esdf={min_esdf:.3f} < {esdf_safety_threshold})")
                            print(f"🔄 应用旋转避障策略...")

            # 9. 如果需要旋转，应用基于旋转的避障
            # 9. If rotation is needed, apply rotation-based obstacle avoidance
            if need_rotation:
                # 修改最佳轨迹：X 轴设为 0（停止前进），Y 轴保持转向
                # Modify the best trajectory: set the X axis to 0 (stop moving forward) and keep the Y axis for turning
                modified_best = best_trajectory_final.copy()

                # 停止前进（X 轴设为 0）
                # Stop moving forward (set the X axis to 0)
                modified_best[:, :, 0] = 0.0  # X 轴：前进方向设为 0 / X axis: set the forward direction to 0

                # 转向方式：取轨迹 Y 方向平均值的符号
                # 计算轨迹 Y 方向的平均值
                # Turning: take the sign of the trajectory mean along Y
                # Compute the average of the trajectory's Y component
                avg_y = modified_best[:, :, 1].max()

                if abs(avg_y) < 0.001:  # 如果平均 Y 接近 0，根据目标决定转向 / if the average Y is near 0, decide the turn from the goal
                    # 根据目标位置决定转向方向
                    # Decide the turning direction from the goal position
                    if len(goals) > 0:
                        goal_y = goals[0][1]  # 目标的 Y 坐标 / the goal's Y coordinate
                        turn_sign = 1.0 if goal_y > 0 else -1.0
                    else:
                        turn_sign = 1.0  # 默认右转 / default to turning right
                    modified_best[:, :, 1] = turn_sign
                else:
                    # 使用平均值的符号，整个轨迹设为统一的转向
                    # Use the sign of the average and apply a uniform turn to the whole trajectory
                    turn_sign = np.sign(avg_y)
                    modified_best[:, :, 1] = turn_sign

                best_trajectory_final = modified_best

                # 同样修改 all_trajectory 中的第一个轨迹（最佳轨迹）
                # Likewise modify the first trajectory in all_trajectory (the best trajectory)
                if all_trajectory_final.shape[1] > 0:
                    all_trajectory_final[:, 0, :, :] = modified_best[0, :, :]  # 修改第一个轨迹 / modify the first trajectory

                if self.verbose:
                    print(f"✅ 旋转避障完成:")
                    print(f"   - X轴(前进)已停止: {modified_best[0, :5, 0].tolist()}")  # 显示前 5 个点的 X 值 / show the X values of the first 5 points
                    print(f"   - Y轴(转向)已调整: {modified_best[0, :5, 1].tolist()}")  # 显示前 5 个点的 Y 值 / show the Y values of the first 5 points

            if self.verbose:
                print(f"✅ 返回结果 (标准格式):")
                print(f"   - execute_trajectory: {best_trajectory_final.shape}")
                print(f"   - all_trajectory: {all_trajectory_final.shape}")
                print(f"   - all_values: {values_final.shape if values_final is not None else None}")
                if values_final is not None and len(values_final) > 0:
                    print(f"     * 最佳cost: {values_final[0, 0]:.3f}")
                    if values_final.shape[1] > 1:
                        print(f"     * 最差cost: {values_final[0, -1]:.3f}")
                        print(f"     * cost范围: [{values_final.min():.3f}, {values_final.max():.3f}]")

                    # 验证 execute_trajectory（最佳轨迹）对应最小 cost
                    # Verify that execute_trajectory (the best trajectory) corresponds to the minimum cost
                    print(f"\n🎯 最终验证:")
                    print(f"   - execute_trajectory (best) cost: {values_final[0, 0]:.3f}")
                    print(f"   - all_trajectory[0] (first) cost: {values_final[0, 0]:.3f}")

                    # 正确的形状对比：best_trajectory_final 是 (1, L, 3)，all_trajectory_final[:, 0, :, :] 是 (1, L, 3)
                    # Shape-consistent comparison: best_trajectory_final is (1, L, 3), all_trajectory_final[:, 0, :, :] is (1, L, 3)
                    first_traj_from_all = all_trajectory_final[:, 0, :, :]  # (1, L, 3)
                    is_same = np.array_equal(best_trajectory_final, first_traj_from_all)

                    print(f"   - execute_trajectory == all_trajectory[0]: {'✅' if is_same else '❌'}")

                    if not is_same and need_rotation:
                        print(f"      ⚠️ 注意：因为应用了旋转避障，execute_trajectory被修改了")
                        print(f"      这是正常的！旋转避障会修改best_trajectory但不修改all_trajectory")
                    elif not is_same:
                        print(f"      ⚠️ 轨迹不匹配且未触发旋转避障 / trajectory mismatch without rotation-based avoidance")
                        # 调试信息
                        # Debug information
                        print(f"         best shape: {best_trajectory_final.shape}")
                        print(f"         all[0] shape: {first_traj_from_all.shape}")
                        print(f"         差异: {np.abs(best_trajectory_final - first_traj_from_all).max():.6f}")

                    # 检查 values_final 是否已排序
                    # Check whether values_final is sorted
                    if values_final.shape[1] > 1:
                        is_sorted = all(values_final[0, i] <= values_final[0, i+1] for i in range(values_final.shape[1]-1))
                        print(f"   - values_final已排序（cost从小到大）: {'✅ 正确' if is_sorted else '❌ 错误！'}")
                        if not is_sorted:
                            print(f"      前5个值: {values_final[0, :5]}")
                if need_rotation:
                    print(f"     * 已应用旋转避障策略")

            return best_trajectory_final, all_trajectory_final, values_final, trajectory_mask

        except Exception as e:
            if self.verbose:
                print(f"❌ SanD-planner推理失败: {e}")
                print(f"   错误类型: {type(e).__name__}")
                if "index" in str(e).lower():
                    print("   这是一个索引错误，可能是由于轨迹数组形状问题导致的")
                import traceback
                traceback.print_exc()

                print("🔄 使用默认安全轨迹...")
            # 返回默认值，确保输出格式正确
            # Return default values, ensuring the output format is correct
            best_trajectory, all_trajectories, values = self._generate_default_trajectory(batch_size=1, num_trajectories=16)

            # 生成默认 trajectory_mask
            # Generate a default trajectory_mask
            try:
                if images is not None:
                    trajectories_for_viz = all_trajectories[np.newaxis, :]  # 添加 batch 维度 / add the batch dimension
                    values_for_viz = values[np.newaxis, :]  # 添加 batch 维度 / add the batch dimension
                    trajectory_mask = self.project_trajectory(images, trajectories_for_viz, values_for_viz)
                else:
                    trajectory_mask = np.zeros((1, 224, 224, 3))
            except Exception as viz_error:
                print(f"⚠️ 可视化也失败: {viz_error}")
                trajectory_mask = images.copy() if images is not None else np.zeros((1, 224, 224, 3))

            return best_trajectory, all_trajectories, values, trajectory_mask



    def project_trajectory(self, images, n_trajectories, n_values):
        """将轨迹投影到图像平面 / Project trajectories onto the image plane."""
        # 导入必要的 colormap
        # Import the required colormap utilities
        from matplotlib import colormaps as cm

        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i, :, :, 0:2]  # 只取 x, y 坐标 / take only the x, y coordinates
            n_value = n_values[i] if n_values is not None else np.zeros(n_trajectories.shape[1])

            for waypoints, value in zip(n_trajectory, n_value):
                norm_value = np.clip(-value * 0.1, 0, 1)
                colormap = cm.get_cmap('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0], 3)) - 0.2
                input_points[:, 0:2] = waypoints
                input_points[:, 1] = -input_points[:, 1]

                # 使用相机内参投影到图像坐标
                # Project to image coordinates using the camera intrinsics
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:, 2] / (input_points[:, 0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:, 1] / (input_points[:, 0] + 1e-8) + self.image_intrinsic[0][2]

                # 绘制轨迹线段
                # Draw the trajectory line segments
                for j in range(camera_x.shape[0] - 1):
                    try:
                        if camera_x[j] > 0 and camera_z[j] > 0 and camera_x[j+1] > 0 and camera_z[j+1] > 0:
                            if cv2 is not None:
                                trajectory_mask = cv2.line(
                                    trajectory_mask,
                                    (int(camera_x[j]), int(camera_z[j])),
                                    (int(camera_x[j+1]), int(camera_z[j+1])),
                                    color.astype(np.uint8).tolist(),
                                    5
                                )
                    except:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks, axis=1)

    def __del__(self):
        """清理临时目录 / Clean up the temporary directory."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


# 简化的测试函数
# Simplified test function
def test_simple_sand_planner():
    """测试简化版 SanD-planner Agent / Test the simplified SanD-planner Agent."""
    print("🧪 测试简化版SanD-planner Agent")

    # 创建虚拟相机内参
    # Create dummy camera intrinsics
    image_intrinsic = np.array([
        [389.551, 0, 324.211],
        [0, 389.551, 235.656],
        [0, 0, 1]
    ])

    try:
        config = InferenceConfig(
            device='cuda:0',
            save_visualizations=False,
            save_data=False,
        )
        agent = SandPlannerAgent(
            image_intrinsic=image_intrinsic,
            config=config,
        )

        # 重置
        # Reset
        agent.reset(batch_size=1, threshold=0.1)

        # 创建测试数据
        # Create test data
        test_images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        test_depths = np.random.rand(1, 168, 224, 1) * 5.0  # 5 米深度 / 5-meter depth
        test_goals = np.array([[3.0, 1.0, 0.0]])

        # 测试点目标导航
        # Test point-goal navigation
        best_trajectory, all_trajectories, values, mask = agent.step_pointgoal(
            test_goals, test_images, test_depths
        )

        print(f"✅ 测试成功!")
        print(f"   - best_trajectory: {best_trajectory.shape}")
        print(f"   - all_trajectories: {all_trajectories.shape}")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_sand_planner()
