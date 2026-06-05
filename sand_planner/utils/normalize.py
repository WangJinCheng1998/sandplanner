#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_utils: 基于 compute_trajectory_stats.py 生成的 JSON 做按轴归一化/反归一化。 /
normalize_utils: per-axis normalization/denormalization based on the JSON produced by
compute_trajectory_stats.py.

支持两种模式：
- percentile: 使用分位数区间（默认 p1-p99）并向两端各扩展 margin（默认 10%）后映射到 [-1, 1]。
- zscore: 使用均值/标准差做 z-score 归一化（可选夹断）。

Two modes are supported:
- percentile: use a percentile interval (default p1-p99), expand each end by `margin`
  (default 10%), then map to [-1, 1].
- zscore: z-score normalization using mean/std (with optional clamping).

用途：对 B-spline 控制点 (8,3) 与相对终点 (3,) 进行一致的 per-axis 归一化。 /
Purpose: apply consistent per-axis normalization to B-spline control points (8,3) and the
relative endpoint (3,).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Literal, Optional
import numpy as np


class AxisScaler:
    """单轴缩放器，提供 normalize/denormalize。 / Single-axis scaler providing normalize/denormalize."""

    def __init__(self,
                 method: Literal["percentile", "zscore"],
                 params: Dict,
                 clamp: bool = True):
        self.method = method
        self.params = params
        self.clamp = clamp

        if method == "percentile":
            lo = float(params["lo"])
            hi = float(params["hi"])
            if hi <= lo:
                # 防御：避免除零 / Guard: avoid division by zero
                hi = lo + 1e-6
            self.mid = (lo + hi) * 0.5
            self.half_range = (hi - lo) * 0.5
        elif method == "zscore":
            self.mean = float(params.get("mean", 0.0))
            self.std = float(params.get("std", 1.0))
            if self.std <= 0:
                self.std = 1.0
            # 可选：用于夹断的阈值（基于分位数换算成 z）
            # Optional: clamping threshold (a percentile converted to a z value)
            self.z_clip = float(params.get("z_clip", 0.0))  # 0 表示不夹断 / 0 means no clamping
        else:
            raise ValueError(f"Unknown method: {method}")

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.method == "percentile":
            y = (x - self.mid) / self.half_range
            if self.clamp:
                y = np.clip(y, -1.0, 1.0)
            return y
        else:  # zscore
            y = (x - self.mean) / self.std
            if self.clamp and self.params.get("z_clip", 0.0) > 0:
                zc = float(self.params["z_clip"])
                y = np.clip(y, -zc, zc)
            return y

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        if self.method == "percentile":
            x = y * self.half_range + self.mid
            return x
        else:  # zscore
            x = y * self.std + self.mean
            return x


class TrajectoryNormalizer:
    """
    三轴归一化器。输入/输出均支持形状 (..., 3) 的数组（最后一个维度为 [x,y,z]）。 /
    Three-axis normalizer. Both input and output accept arrays of shape (..., 3),
    where the last dimension is [x, y, z].
    """

    def __init__(self,
                 stats_json_path: str,
                 method: Literal["percentile", "zscore"] = "percentile",
                 margin: float = 0.10,
                 use_percentiles: tuple = (1,99),
                 clamp: bool = True):
        self.stats_json_path = stats_json_path
        self.method = method
        self.margin = float(margin)
        self.clamp = clamp
        with open(stats_json_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        axis_stats: Dict[str, Dict] = self.stats.get("axis", {})
        self.scalers = []

        if method == "percentile":
            p_low, p_high = use_percentiles
            for axis_name in ["x", "y", "z"]:
                s = axis_stats.get(axis_name, {})
                lo = float(s.get(f"p{p_low}", s.get("min", -1.0)))
                hi = float(s.get(f"p{p_high}", s.get("max", 1.0)))
                # 对称地向两端扩展 margin 余量（默认 10%）
                # Expand the safety margin symmetrically at both ends (default 10%)
                span = hi - lo
                expand = self.margin * span
                lo_e = lo - expand
                hi_e = hi + expand
                params = {"lo": lo_e, "hi": hi_e}
                self.scalers.append(AxisScaler("percentile", params, clamp=clamp))

        elif method == "zscore":
            # zscore：使用 mean/std，可选通过 p 值换算得到 z_clip，实现“带余量的夹断”
            # zscore: use mean/std; optionally convert percentiles into a z_clip to obtain
            # a clamping threshold with a safety margin
            for axis_name in ["x", "y", "z"]:
                s = axis_stats.get(axis_name, {})
                mean = float(s.get("mean", 0.0))
                std = float(s.get("std", 1.0))
                # 通过分位数估计一个 z_clip，再放宽 margin（默认 10%）
                # Estimate a z_clip from percentiles, then relax it by `margin` (default 10%)
                p_low, p_high = use_percentiles
                lo = float(s.get(f"p{p_low}", mean - 3*std))
                hi = float(s.get(f"p{p_high}", mean + 3*std))
                # 估算对应的 z 范围 / Estimate the corresponding z range
                z_lo = (lo - mean) / max(std, 1e-6)
                z_hi = (hi - mean) / max(std, 1e-6)
                z_clip = max(abs(z_lo), abs(z_hi)) * (1 + self.margin)
                params = {"mean": mean, "std": std, "z_clip": z_clip}
                self.scalers.append(AxisScaler("zscore", params, clamp=clamp))
        else:
            raise ValueError(f"Unknown method: {method}")

    def normalize(self, xyz: np.ndarray) -> np.ndarray:
        arr = np.asarray(xyz, dtype=np.float32)
        assert arr.shape[-1] == 3, "最后一维必须是3 (x,y,z)"
        out = arr.copy()
        for i in range(3):
            out[..., i] = self.scalers[i].normalize(out[..., i])
        return out

    def denormalize(self, xyz_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(xyz_norm, dtype=np.float32)
        assert arr.shape[-1] == 3, "最后一维必须是3 (x,y,z)"
        out = arr.copy()
        for i in range(3):
            out[..., i] = self.scalers[i].denormalize(out[..., i])
        return out
    
    def normalize_batch(self, xyz_batch: np.ndarray) -> np.ndarray:
        """批次归一化，支持形状 [B, N, 3] 的数组。 / Batch normalization for arrays of shape [B, N, 3]."""
        return self.normalize(xyz_batch)
    
    def denormalize_batch(self, xyz_norm_batch: np.ndarray) -> np.ndarray:
        """批次反归一化，支持形状 [B, N, 3] 的数组。 / Batch denormalization for arrays of shape [B, N, 3]."""
        return self.denormalize(xyz_norm_batch)

    def normalized_zero(self) -> np.ndarray:
        """返回 (0,0,0) 在归一化空间中的值（便于需要固定首控制点的场景）。 /
        Return the value of (0,0,0) in normalized space (useful when the first control point
        must be fixed)."""
        zero = np.zeros(3, dtype=np.float32)
        return self.normalize(zero)
    
    def normalize_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        将世界坐标系中的梯度转换为归一化空间中的梯度。 /
        Convert gradients in the world coordinate frame into normalized space.

        Args:
            gradients: [..., 3] 世界坐标系中的梯度 / gradients in the world coordinate frame.

        Returns:
            normalized_gradients: [..., 3] 归一化空间中的梯度 / gradients in normalized space.
        """
        arr = np.asarray(gradients, dtype=np.float32)
        assert arr.shape[-1] == 3, "最后一维必须是3 (x,y,z)"
        out = arr.copy()
        
        for i in range(3):
            scaler = self.scalers[i]
            if scaler.method == "percentile":
                # 分位数方法：梯度需除以半范围 / percentile method: divide the gradient by the half-range
                out[..., i] = out[..., i] / scaler.half_range
            elif scaler.method == "zscore":
                # z-score 方法：梯度需除以标准差 / z-score method: divide the gradient by the std
                out[..., i] = out[..., i] / scaler.std
                
        return out
    
    def denormalize_gradients(self, normalized_gradients: np.ndarray) -> np.ndarray:
        """
        将归一化空间中的梯度转换为世界坐标系中的梯度。 /
        Convert gradients in normalized space into the world coordinate frame.

        Args:
            normalized_gradients: [..., 3] 归一化空间中的梯度 / gradients in normalized space.

        Returns:
            gradients: [..., 3] 世界坐标系中的梯度 / gradients in the world coordinate frame.
        """
        arr = np.asarray(normalized_gradients, dtype=np.float32)
        assert arr.shape[-1] == 3, "最后一维必须是3 (x,y,z)"
        out = arr.copy()
        
        for i in range(3):
            scaler = self.scalers[i]
            if scaler.method == "percentile":
                # 分位数方法：梯度需乘以半范围 / percentile method: multiply the gradient by the half-range
                out[..., i] = out[..., i] * scaler.half_range
            elif scaler.method == "zscore":
                # z-score 方法：梯度需乘以标准差 / z-score method: multiply the gradient by the std
                out[..., i] = out[..., i] * scaler.std
                
        return out


def load_normalizer(stats_json_path: str,
                    method: Literal["percentile", "zscore"] = "percentile",
                    margin: float = 0.10,
                    clamp: bool = True) -> TrajectoryNormalizer:
    return TrajectoryNormalizer(stats_json_path, method=method, margin=margin, clamp=clamp)
