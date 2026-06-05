#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SanD-planner 的深度图处理器。
Depth processors for SanD-planner.

FileDepthProcessor  -- 基于文件的深度图加载（独立推理） / file-based depth loading (standalone inference)
ArrayDepthProcessor -- 基于数组的深度图处理，支持特征缓存（运行时） / array-based depth processing with feature caching (runtime)
"""

import os

import cv2
import numpy as np
import torch

from sand_planner.config import InferenceConfig
from sand_planner.utils.image import downscale_to_target_size


# ---------------------------------------------------------------------------
# FileDepthProcessor
# ---------------------------------------------------------------------------

class FileDepthProcessor:
    """深度图像处理器（基于文件） / Depth image processor (file-based)."""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def load_for_model(self, depth_path: str) -> np.ndarray:
        """加载并预处理深度图像用于模型输入 / Load and preprocess a depth image for model input."""
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise ValueError(f"无法读取深度图像: {depth_path}")

        # 转换为米并裁剪 / Convert to meters and clip
        depth_array = depth_img.astype(np.float32) / 1000.0
        depth_array = np.clip(depth_array, 0, self.config.max_depth)

        # 下采样 / Downsampling
        if self.config.downscale_depth:
            try:
                depth_array = downscale_to_target_size(
                    depth_array, self.config.image_height, self.config.image_width, is_depth=True
                )
            except Exception:
                pass

        # 调整大小 / Resize
        target_size = (self.config.image_height, self.config.image_width)
        if depth_array.shape[:2] != target_size:
            depth_array = cv2.resize(
                depth_array, (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_NEAREST
            )

        # 归一化到 [0, 1] / Normalize to [0, 1]
        return depth_array / self.config.max_depth

    def load_for_visualization(self, depth_path: str) -> np.ndarray:
        """加载深度图像用于可视化，保持原始分辨率 / Load a depth image for visualization, keeping the original resolution."""
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise ValueError(f"无法读取深度图像: {depth_path}")

        depth_array = depth_img.astype(np.float32) / 1000.0
        depth_array = np.clip(depth_array, 0, self.config.max_depth)
        return depth_array / self.config.max_depth



    def create_sequence(self, depth: np.ndarray) -> torch.Tensor:
        """创建深度序列（重复模式） / Create a depth sequence (repeat mode)."""
        depths = np.stack([depth] * self.config.sequence_length, axis=0)
        return torch.from_numpy(depths).float().unsqueeze(1)

    def load_consecutive_sequence(self, start_frame: int = 1) -> torch.Tensor:
        """加载连续深度序列 / Load a sequence of consecutive depth frames."""
        depths = []
        for i in range(self.config.sequence_length):
            frame_idx = start_frame + i
            depth_filename = f"depth_{frame_idx:04d}.png"
            depth_path = os.path.join(self.config.consecutive_depth_dir, depth_filename)

            if os.path.exists(depth_path):
                depth_array = self.load_for_model(depth_path)
                depths.append(depth_array)
            elif depths:
                depths.append(depths[-1])  # 重复最后一帧 / Repeat the last frame
            else:
                raise FileNotFoundError(f"无法找到起始深度图: {depth_path}")

        depths = np.stack(depths, axis=0)
        return torch.from_numpy(depths).float().unsqueeze(1)


# ---------------------------------------------------------------------------
# ArrayDepthProcessor
# ---------------------------------------------------------------------------

class ArrayDepthProcessor:
    """深度图像处理器（基于数组），移入 agent 以提升性能，支持特征缓存 / Depth image processor (array-based), moved into the agent for performance, with feature caching."""

    def __init__(self, config):
        self.config = config
        # 缓存编码后的特征以提高效率 / Cache encoded features to improve efficiency
        self.encoded_features_cache = []  # 存储编码后的特征 List[torch.Tensor] / Stores encoded features, List[torch.Tensor]
        self.max_cache_size = config.depth_cache_size  # 最大缓存帧数 / Maximum number of cached frames

    def process_depth_array_for_model(self, depth_array) -> np.ndarray:
        """直接处理深度数组用于模型输入，避免文件 IO / Process a depth array directly for model input, avoiding file IO."""
        import numpy as np

        # 确保是浮点类型 / Ensure float32 dtype
        if depth_array.dtype != np.float32:
            depth_array = depth_array.astype(np.float32)

        # 若输入以米为单位则直接使用；若为毫米则转换为米
        # If the input is in meters, use it directly; if in millimeters, convert to meters
        if depth_array.max() > 50:  # 假设大于 50 的是毫米单位 / Assume values above 50 are in millimeters
            depth_array = depth_array / 1000.0

        # 裁剪深度值 / Clip depth values
        depth_array = np.clip(depth_array, 0, self.config.max_depth)

        # 下采样（如果启用） / Downsampling (if enabled)
        if self.config.downscale_depth:
            try:
                from sand_planner.utils.image import downscale_to_target_size
                depth_array = downscale_to_target_size(
                    depth_array, self.config.image_height, self.config.image_width, is_depth=True
                )
            except ImportError:
                pass  # 如果 import 失败，跳过下采样 / If the import fails, skip downsampling

        # 调整大小到目标尺寸 / Resize to the target size
        target_size = (self.config.image_height, self.config.image_width)
        if len(depth_array.shape) == 3:  # (H, W, 1)
            depth_array = depth_array[:, :, 0]  # 移除通道维度 / Remove the channel dimension

        if depth_array.shape != target_size:
            if cv2 is not None:
                depth_array = cv2.resize(
                    depth_array, (self.config.image_width, self.config.image_height),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                # 如果 cv2 不可用，使用 numpy 简单插值
                # If cv2 is unavailable, fall back to simple numpy interpolation
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = (target_size[0] / depth_array.shape[0],
                                   target_size[1] / depth_array.shape[1])
                    depth_array = zoom(depth_array, zoom_factors, order=0)
                except ImportError:
                    # 如果 scipy 也不可用，使用简单的重复采样
                    # If scipy is also unavailable, fall back to simple repeat sampling
                    import numpy as np
                    h_old, w_old = depth_array.shape
                    h_new, w_new = target_size
                    depth_array = np.repeat(np.repeat(depth_array, h_new//h_old, axis=0)[:h_new], w_new//w_old, axis=1)[:, :w_new]

        # 归一化到 [0, 1] / Normalize to [0, 1]
        return depth_array / self.config.max_depth

    def encode_single_depth_frame(self, depth_array: np.ndarray) -> torch.Tensor:
        """编码单帧深度图像为特征，用于缓存 / Encode a single depth frame into features for caching.

        Args:
            depth_array: (H, W, 1) 或 (H, W) 单帧深度数组 / a single-frame depth array of shape (H, W, 1) or (H, W).

        Returns:
            torch.Tensor: (1, H, W) 编码后的单帧特征 / (1, H, W) encoded single-frame features.
        """
        # 处理深度数组 / Process the depth array
        processed_depth = self.process_depth_array_for_model(depth_array)

        # 转换为张量并添加 batch 和 channel 维度 / Convert to a tensor and add batch and channel dimensions
        depth_tensor = torch.from_numpy(processed_depth).float().unsqueeze(0)  # (1, H, W)

        return depth_tensor

    def create_sequence_from_array(self, depth: np.ndarray) -> torch.Tensor:
        """从深度数组创建序列（重复模式） / Create a sequence from a depth array (repeat mode)."""
        depths = np.stack([depth] * self.config.sequence_length, axis=0)
        return torch.from_numpy(depths).float().unsqueeze(1)  # (seq_len, 1, H, W)

    def encode_depth_for_model(self, depth_arrays: np.ndarray) -> torch.Tensor:
        """编码深度数组用于模型推理 / Encode depth arrays for model inference.

        Args:
            depth_arrays: (batch_size, H, W, 1) 或 (batch_size, H, W) 深度数组 / depth arrays of shape (batch_size, H, W, 1) or (batch_size, H, W).

        Returns:
            torch.Tensor: (batch_size, seq_len, 1, H, W) 编码后的深度序列 / (batch_size, seq_len, 1, H, W) encoded depth sequences.
        """
        batch_sequences = []

        for i in range(depth_arrays.shape[0]):
            # 获取单个深度图 / Get a single depth image
            if len(depth_arrays.shape) == 4:  # (batch_size, H, W, 1)
                depth_img = depth_arrays[i, :, :, 0]  # 移除最后一个维度 / Remove the last dimension
            else:  # (batch_size, H, W)
                depth_img = depth_arrays[i]

            # 处理深度图 / Process the depth image
            processed_depth = self.process_depth_array_for_model(depth_img)

            # 创建序列 / Create the sequence
            depth_sequence = self.create_sequence_from_array(processed_depth)

            batch_sequences.append(depth_sequence)

        # 堆叠成批次 (batch_size, seq_len, 1, H, W) / Stack into a batch of shape (batch_size, seq_len, 1, H, W)
        return torch.stack(batch_sequences, dim=0)

    def add_frame_to_cache(self, depth_array: np.ndarray, should_save: bool = True) -> None:
        """添加新帧到特征缓存（支持跳帧） / Add a new frame to the feature cache (supports frame skipping).

        Args:
            depth_array: (H, W, 1) 或 (H, W) 新的深度帧 / a new depth frame of shape (H, W, 1) or (H, W).
            should_save: 是否应该保存此帧到历史缓存（用于跳帧逻辑） / whether to save this frame into the history cache (for the frame-skipping logic).
        """
        # 编码单帧 / Encode the single frame
        encoded_frame = self.encode_single_depth_frame(depth_array)

        # 如果 should_save 为 False，不添加到历史缓存（实现跳帧）
        # If should_save is False, do not add it to the history cache (implements frame skipping)
        if not should_save:
            # 跳过此帧，不保存到历史；但仍然编码了特征（可能用于其他用途）
            # Skip this frame and do not save it to history; the features are still encoded (may be used elsewhere)
            return

        # 添加到缓存 / Add to the cache
        self.encoded_features_cache.append(encoded_frame)

        # 保持缓存大小限制 / Enforce the cache size limit
        if len(self.encoded_features_cache) > self.max_cache_size:
            self.encoded_features_cache.pop(0)  # 移除最旧的特征 / Remove the oldest feature

    def get_sequence_from_cache(self) -> torch.Tensor:
        """从缓存中构建深度序列 / Build a depth sequence from the cache.

        Returns:
            torch.Tensor: (1, seq_len, 1, H, W) 深度序列 / (1, seq_len, 1, H, W) depth sequence.
        """
        if not self.encoded_features_cache:
            raise ValueError("特征缓存为空，请先添加帧")

        # 如果缓存帧数不足，用最新帧重复填充 / If there are too few cached frames, pad by repeating the latest frame
        cached_features = self.encoded_features_cache.copy()
        while len(cached_features) < self.config.sequence_length:
            cached_features.append(cached_features[-1])  # 重复最新帧 / Repeat the latest frame

        # 取最近的 sequence_length 帧 / Take the most recent sequence_length frames
        recent_features = cached_features[-self.config.sequence_length:]

        # 堆叠成序列 (seq_len, 1, H, W) / Stack into a sequence of shape (seq_len, 1, H, W)
        sequence = torch.stack(recent_features, dim=0)

        # 添加 batch 维度 (1, seq_len, 1, H, W) / Add the batch dimension -> (1, seq_len, 1, H, W)
        return sequence.unsqueeze(0)

    def clear_cache(self):
        """清空特征缓存 / Clear the feature cache."""
        self.encoded_features_cache.clear()

    def get_cache_info(self) -> dict:
        """获取缓存信息 / Get cache information."""
        return {
            'cache_size': len(self.encoded_features_cache),
            'max_cache_size': self.max_cache_size,
            'is_full': len(self.encoded_features_cache) >= self.config.sequence_length
        }
