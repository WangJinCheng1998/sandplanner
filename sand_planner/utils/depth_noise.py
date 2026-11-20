"""
仿 RealSense D435 的深度噪声模拟器 / RealSense D435-style depth noise simulator.

把噪声逻辑集中在一个模块中，供训练 dataloader（可选的深度噪声增强）和可视化工具复用。
训练时使用的是 V3 版本（D435NoiseConfigV3 / D435DepthNoiseSimulatorV3）；V1、V2 保留作参考。
Centralizes the noise logic so it can be reused by the training dataloader (optional depth-noise
augmentation) and by visualization tools. Training uses V3 (D435NoiseConfigV3 /
D435DepthNoiseSimulatorV3); V1 and V2 are kept for reference.

轴向噪声的二次多项式系数取自 / The axial-noise quadratic coefficients follow:
    Ahn et al., "Analysis and Noise Modeling of the Intel RealSense D435 for Mobile Robots", 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import math

import numpy as np
import cv2

ArrayLike = np.ndarray


def _generate_low_frequency_noise(
    shape: Tuple[int, int], block: int, rng: np.random.Generator
) -> ArrayLike:
    """生成平滑的低频噪声场（双线性插值上采样，避免方块状伪影）。 / Generate a smooth low-frequency noise field (bilinear upsampling avoids blocky artifacts)."""
    h, w = shape
    block_h = max(1, math.ceil(h / block))
    block_w = max(1, math.ceil(w / block))

    # 生成低分辨率噪声 / Generate low-resolution noise
    coarse = rng.normal(0.0, 1.0, size=(block_h, block_w)).astype(np.float32)

    # 双线性插值上采样到目标分辨率，避免方块状边界 / Bilinearly upsample to the target resolution to avoid blocky borders
    noise = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    # 归一化 / Normalize
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise


def _compute_edge_strength(depth: ArrayLike, valid_mask: ArrayLike) -> ArrayLike:
    """计算归一化的深度梯度幅值（无效像素先用中位数填充）。 / Compute the normalized depth-gradient magnitude (invalid pixels are first filled with the median)."""
    filled = depth.copy()
    if np.any(valid_mask):
        filled[~valid_mask] = np.median(depth[valid_mask])
    else:
        filled[~valid_mask] = 0.0
    gx, gy = np.gradient(filled)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = grad / (grad.max() + 1e-6)
    return grad


def _create_hole_mask(
    edge_strength: ArrayLike,
    depth_norm: ArrayLike,
    rng: np.random.Generator,
    base: float,
    edge_scale: float,
    depth_decay: float,
    cluster_block: int,
    cluster_threshold: float,
) -> ArrayLike:
    """生成成块出现的空洞掩码（V1 使用）。 / Generate a clustered hole mask (used by V1)."""
    h, w = edge_strength.shape
    low_freq = _generate_low_frequency_noise((h, w), block=cluster_block, rng=rng)
    prob = (base + edge_scale * edge_strength) * np.exp(-depth_decay * depth_norm)
    prob = np.clip(prob, 0.0, 0.5)
    mask = rng.random((h, w)) < prob
    mask &= low_freq > cluster_threshold
    return mask


@dataclass
class D435NoiseConfig:
    """V1 配置：高斯 + speckle + 边缘偏移 + 空洞 + dropout。 / V1 config: Gaussian + speckle + edge bias + holes + dropout."""
    gaussian_base: float = 0.001
    gaussian_scale: float = 0.00075
    speckle_base: float = 0.006
    speckle_scale: float = 0.010
    edge_bias_min: float = 0.004
    edge_bias_max: float = 0.009
    hole_base: float = 0.02
    hole_edge_scale: float = 0.12
    hole_depth_decay: float = 1.3
    hole_cluster_block: int = 18
    hole_cluster_threshold: float = 0.8
    dropout_base: float = 0.01
    dropout_depth_decay: float = 2.5
    invalid_fill_mode: str = "zero"  # 可选值/options: {"zero", "maxdepth"}

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class D435NoiseConfigV2:
    """V2 配置：结合 Ahn 2019 的参数与 V1 的实现。 / V2 config: combines the Ahn 2019 parameters with the V1 implementation.

    相对 V1 的改进 / Changes relative to V1:
    - 轴向噪声使用 Ahn 2019 的完整二次多项式 σ(z) = c0 + c1*z + c2*z²
      Axial noise uses the full Ahn 2019 quadratic σ(z) = c0 + c1*z + c2*z²
    - 空洞概率增加随距离线性增长的项 / Hole probability gains a term that grows linearly with range
    - 新增左侧无效带（IR 与 RGB 视场不重合） / Adds a left invalid band (IR and RGB fields of view do not fully overlap)
    - 保留 speckle 空间相关噪声和边缘偏移 / Keeps the spatially correlated speckle noise and the edge bias
    """
    # Ahn 2019 轴向噪声 / Ahn 2019 axial noise: σ(z) = c0 + c1*z + c2*z²
    axial_c0: float = 0.001063      # 常数项（米） / constant term (m)
    axial_c1: float = 0.0007278     # 线性项系数 / linear coefficient
    axial_c2: float = 0.003949      # 二次项系数 / quadratic coefficient
    axial_scale: float = 1.0        # 整体缩放（D435f 可取 0.7~0.9） / overall scale (0.7~0.9 for D435f)

    # Speckle 低频空间相关噪声（IR 投影图案） / Low-frequency, spatially correlated speckle noise (IR projector pattern)
    speckle_base: float = 0.006
    speckle_scale: float = 0.010
    speckle_block: int = 16         # 低频噪声块大小 / low-frequency noise block size

    # 边缘系统偏移（飞点） / Systematic edge bias (flying pixels)
    edge_bias_min: float = 0.004
    edge_bias_max: float = 0.009

    # 空洞（边缘项 + 距离项） / Holes (edge term + range term)
    hole_base: float = 0.02
    hole_edge_scale: float = 0.12   # 边缘梯度权重 / weight of the edge gradient
    hole_range_scale: float = 0.25  # 距离线性项 / linear range term
    hole_depth_decay: float = 1.3   # 指数衰减 / exponential decay
    hole_cluster_block: int = 18
    hole_cluster_threshold: float = 0.8

    # 随机 dropout / Random dropout
    dropout_base: float = 0.01
    dropout_depth_decay: float = 2.5

    # 左侧无效带（IR/RGB 视场差异），固定像素宽度 / Left invalid band (IR/RGB FOV difference), fixed width in pixels
    left_band_enabled: bool = True
    left_band_px: int = 12

    invalid_fill_mode: str = "zero"  # 可选值/options: {"zero", "maxdepth"}

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class D435NoiseConfigV3:
    """V3 配置（训练 dataloader 使用）：在 V2 基础上增加尖刺噪声和边缘缺失。 / V3 config (used by the training dataloader): adds spike noise and edge dropouts on top of V2.

    相对 V2 的新增项 / Additions relative to V2:
    - 尖刺噪声：边缘处的深度突变 / Spiky artifacts: sudden depth jumps at edges
    - 边缘腐蚀：物体边界周围成片的深度缺失 / Edge erosion: contiguous missing depth around object boundaries
    - 深度不连续检测：大跳变处增加无效概率 / Discontinuity check: higher invalid probability at large depth jumps

    默认值下：轴向噪声、speckle、边缘偏移、左侧无效带、尖刺、边缘腐蚀、不连续检测和随机 dropout
    均开启；V2 的空洞项关闭（hole_base = hole_edge_scale = hole_range_scale = 0）。
    With the defaults, axial noise, speckle, edge bias, the left band, spikes, edge erosion,
    the discontinuity check and random dropout are all on; the V2 hole term is off
    (hole_base = hole_edge_scale = hole_range_scale = 0).
    """
    # ========== 继承自 V2 的参数 / Parameters inherited from V2 ==========
    # Ahn 2019 轴向噪声 / Ahn 2019 axial noise
    axial_c0: float = 0.001063
    axial_c1: float = 0.0007278
    axial_c2: float = 0.003949
    axial_scale: float = 1.0

    # Speckle 噪声 / Speckle noise
    speckle_base: float = 0.006
    speckle_scale: float = 0.010
    speckle_block: int = 16

    # 边缘系统偏移 / Systematic edge bias
    edge_bias_min: float = 0.004
    edge_bias_max: float = 0.009

    # 空洞（默认关闭） / Holes (off by default)
    hole_base: float = 0.0
    hole_edge_scale: float = 0.0
    hole_range_scale: float = 0.0
    hole_depth_decay: float = 1.3
    hole_cluster_block: int = 18
    hole_cluster_threshold: float = -1.0

    # 随机 dropout（默认开启，近处约 1%，随距离衰减） / Random dropout (on by default: ~1% at close range, decaying with range)
    dropout_base: float = 0.01
    dropout_depth_decay: float = 2.5

    # 左侧无效带 / Left invalid band
    left_band_enabled: bool = True
    left_band_px: int = 12

    invalid_fill_mode: str = "zero"

    # ========== V3 新增参数 / Parameters added in V3 ==========
    # 尖刺噪声 / Spiky artifacts
    spike_enabled: bool = True
    spike_probability: float = 0.03        # 边缘处产生尖刺的概率 / probability of a spike at an edge pixel
    spike_depth_shift_min: float = 0.10    # 尖刺深度偏移最小值（相对深度） / minimum spike depth shift (relative)
    spike_depth_shift_max: float = 0.25    # 尖刺深度偏移最大值（相对深度） / maximum spike depth shift (relative)
    spike_width: int = 2                   # 尖刺宽度（像素） / spike width (pixels)
    spike_edge_threshold: float = 0.2      # 边缘强度阈值 / edge-strength threshold

    # 边缘腐蚀（扩大物体边界处的空洞） / Edge erosion (enlarges holes along object boundaries)
    edge_erosion_enabled: bool = True
    edge_erosion_kernel: int = 3           # 腐蚀核大小 / erosion kernel size
    edge_erosion_prob: float = 0.9         # 边缘处额外缺失的概率 / extra dropout probability at edges
    edge_erosion_threshold: float = 0.05   # 边缘强度阈值 / edge-strength threshold

    # 深度不连续处理 / Depth discontinuities
    discontinuity_enabled: bool = True
    discontinuity_threshold: float = 0.3   # 深度跳变阈值（米） / depth-jump threshold (m)
    discontinuity_invalidate_prob: float = 0.4  # 跳变处置为无效的概率 / probability of invalidating a jump pixel

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class D435DepthNoiseSimulator:
    """V1：对深度图施加仿 RealSense D435 的噪声。 / V1: applies RealSense D435-inspired noise to depth maps."""

    def __init__(self, config: Optional[D435NoiseConfig] = None):
        self.config = config or D435NoiseConfig()

    def simulate(
        self,
        depth_m: ArrayLike,
        max_depth: float,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[ArrayLike, ArrayLike, Dict[str, float]]:
        """返回 (加噪深度, 无效掩码, 统计信息)。 / Return (noisy_depth, invalid_mask, stats)."""

        if rng is None:
            rng = np.random.default_rng(seed)
        elif seed is not None:
            # 另建一个生成器，避免改动调用方 rng 的状态 / Derive a throwaway generator to avoid mutating the caller's rng state
            rng = np.random.default_rng(seed)

        depth = depth_m.astype(np.float32)
        valid = depth > 0
        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = np.clip(depth[valid] / max_depth, 0.0, 1.0)

        cfg = self.config

        # 高斯噪声 / Gaussian noise
        sigma = cfg.gaussian_base + cfg.gaussian_scale * np.square(np.clip(depth, 0, max_depth))
        gaussian_noise = rng.normal(0.0, 1.0, size=depth.shape) * sigma * valid

        # Speckle 噪声 / Speckle noise
        speckle_field = _generate_low_frequency_noise(depth.shape, block=16, rng=rng)
        speckle_gain = (cfg.speckle_base + cfg.speckle_scale * depth_norm)
        speckle_noise = speckle_gain * speckle_field * depth

        # 边缘偏移 / Edge bias
        edge_strength = _compute_edge_strength(depth, valid)
        edge_bias_scale = rng.uniform(cfg.edge_bias_min, cfg.edge_bias_max)
        edge_bias = edge_strength * edge_bias_scale * max_depth

        # 空洞 + dropout / Holes + dropouts
        hole_mask = _create_hole_mask(
            edge_strength,
            depth_norm,
            rng,
            cfg.hole_base,
            cfg.hole_edge_scale,
            cfg.hole_depth_decay,
            cfg.hole_cluster_block,
            cfg.hole_cluster_threshold,
        )
        dropout_prob = cfg.dropout_base * np.exp(-cfg.dropout_depth_decay * depth_norm)
        dropout_mask = rng.random(depth.shape) < dropout_prob

        noisy = depth + gaussian_noise + speckle_noise + edge_bias
        noisy = np.clip(noisy, 0.0, max_depth)

        invalid_mask = hole_mask | dropout_mask
        if cfg.invalid_fill_mode == "maxdepth":
            noisy[invalid_mask] = max_depth
        else:
            noisy[invalid_mask] = 0.0

        stats = {
            "gaussian_mm_std": float(np.mean(sigma[valid]) * 1000.0) if np.any(valid) else 0.0,
            "speckle_mm_std": float(np.sqrt(np.mean(np.square(speckle_noise[valid]))) * 1000.0)
            if np.any(valid)
            else 0.0,
            "hole_ratio": float(np.mean(hole_mask)),
            "dropout_ratio": float(np.mean(dropout_mask)),
            "invalid_ratio": float(np.mean(invalid_mask)),
            "edge_bias_cm": float(np.mean(edge_bias[edge_strength > 0.3]) * 100.0)
            if np.any(edge_strength > 0.3)
            else 0.0,
        }

        return noisy.astype(np.float32), invalid_mask, stats


class D435DepthNoiseSimulatorV2:
    """V2：结合 Ahn 2019 参数的 D435 噪声模拟器。 / V2: D435 noise simulator using the Ahn 2019 parameters.

    相对 V1 / Relative to V1:
    - 轴向噪声使用 Ahn 2019 的完整二次多项式 / Axial noise uses the full Ahn 2019 quadratic
    - 空洞概率结合边缘项与距离线性项 / Hole probability combines an edge term and a linear range term
    - 新增左侧无效带 / Adds the left invalid band
    - 保留 speckle 空间相关噪声和边缘偏移 / Keeps the spatially correlated speckle noise and the edge bias
    """

    def __init__(self, config: Optional[D435NoiseConfigV2] = None):
        self.config = config or D435NoiseConfigV2()

    def simulate(
        self,
        depth_m: ArrayLike,
        max_depth: float,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[ArrayLike, ArrayLike, Dict[str, float]]:
        """返回 (加噪深度, 无效掩码, 统计信息)。 / Return (noisy_depth, invalid_mask, stats)."""

        if rng is None:
            rng = np.random.default_rng(seed)
        elif seed is not None:
            rng = np.random.default_rng(seed)

        depth = depth_m.astype(np.float32)
        h, w = depth.shape
        cfg = self.config

        # --------------------------------------------------
        # ⓪ 左侧无效带（IR/RGB 视场差异），最先处理
        #    Left invalid band (IR/RGB FOV difference), applied first
        # --------------------------------------------------
        band_cols = 0
        if cfg.left_band_enabled:
            band_cols = cfg.left_band_px
            depth[:, :band_cols] = 0.0  # 先置为无效，后续噪声在此基础上叠加 / mark invalid first; later noise builds on this

        valid = depth > 0
        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = np.clip(depth[valid] / max_depth, 0.0, 1.0)

        # --------------------------------------------------
        # ① Ahn 2019 轴向噪声 / Ahn 2019 axial noise: σ(z) = c0 + c1*z + c2*z²
        # --------------------------------------------------
        z_clipped = np.clip(depth, 0, max_depth)
        sigma_axial = (cfg.axial_c0 + cfg.axial_c1 * z_clipped + cfg.axial_c2 * z_clipped ** 2)
        sigma_axial = sigma_axial * cfg.axial_scale
        axial_noise = rng.normal(0.0, 1.0, size=depth.shape) * sigma_axial * valid

        # --------------------------------------------------
        # ② Speckle 低频空间相关噪声 / Low-frequency, spatially correlated speckle noise
        # --------------------------------------------------
        speckle_field = _generate_low_frequency_noise(depth.shape, block=cfg.speckle_block, rng=rng)
        speckle_gain = (cfg.speckle_base + cfg.speckle_scale * depth_norm)
        speckle_noise = speckle_gain * speckle_field * depth

        # --------------------------------------------------
        # ③ 边缘系统偏移（飞点） / Systematic edge bias (flying pixels)
        # --------------------------------------------------
        edge_strength = _compute_edge_strength(depth, valid)
        edge_bias_scale = rng.uniform(cfg.edge_bias_min, cfg.edge_bias_max)
        edge_bias = edge_strength * edge_bias_scale * max_depth

        # --------------------------------------------------
        # ④ 空洞（边缘项 + 距离项） / Holes (edge term + range term)
        # --------------------------------------------------
        # 基础概率 + 边缘项 + 距离线性项，再乘以指数衰减
        # Base probability + edge term + linear range term, times an exponential decay
        hole_prob = (cfg.hole_base
                     + cfg.hole_edge_scale * edge_strength
                     + cfg.hole_range_scale * depth_norm)
        hole_prob = hole_prob * np.exp(-cfg.hole_depth_decay * depth_norm)
        hole_prob = np.clip(hole_prob, 0.0, 0.5)

        # 用低频噪声场让空洞成块出现 / A low-frequency field makes holes appear in clusters
        low_freq_cluster = _generate_low_frequency_noise((h, w), block=cfg.hole_cluster_block, rng=rng)
        hole_mask = (rng.random((h, w)) < hole_prob) & (low_freq_cluster > cfg.hole_cluster_threshold)

        # --------------------------------------------------
        # ⑤ 随机 dropout（随距离衰减） / Random dropout (decays with range)
        # --------------------------------------------------
        dropout_prob = cfg.dropout_base * np.exp(-cfg.dropout_depth_decay * depth_norm)
        dropout_mask = rng.random(depth.shape) < dropout_prob

        # --------------------------------------------------
        # ⑥ 叠加噪声 / Combine the noise terms
        # --------------------------------------------------
        noisy = depth + axial_noise + speckle_noise + edge_bias
        noisy = np.clip(noisy, 0.0, max_depth)

        # 合并无效掩码 / Merge the invalid masks
        invalid_mask = hole_mask | dropout_mask

        # 填充无效值 / Fill invalid pixels
        if cfg.invalid_fill_mode == "maxdepth":
            noisy[invalid_mask] = max_depth
        else:
            noisy[invalid_mask] = 0.0

        # 确保左侧无效带包含在最终掩码中 / Make sure the left band is in the final mask
        if cfg.left_band_enabled:
            invalid_mask[:, :band_cols] = True

        # --------------------------------------------------
        # 统计信息 / Statistics
        # --------------------------------------------------
        stats = {
            "axial_mm_std": float(np.mean(sigma_axial[valid]) * 1000.0) if np.any(valid) else 0.0,
            "speckle_mm_std": float(np.sqrt(np.mean(np.square(speckle_noise[valid]))) * 1000.0)
            if np.any(valid) else 0.0,
            "hole_ratio": float(np.mean(hole_mask)),
            "dropout_ratio": float(np.mean(dropout_mask)),
            "invalid_ratio": float(np.mean(invalid_mask)),
            "edge_bias_cm": float(np.mean(edge_bias[edge_strength > 0.3]) * 100.0)
            if np.any(edge_strength > 0.3) else 0.0,
            "left_band_px": band_cols if cfg.left_band_enabled else 0,
        }

        return noisy.astype(np.float32), invalid_mask, stats


class D435DepthNoiseSimulatorV3:
    """V3（训练 dataloader 使用）：在 V2 基础上增加边缘缺失和尖刺噪声。 / V3 (used by the training dataloader): adds edge dropouts and spike noise on top of V2.

    新增 / Additions:
    - 尖刺噪声：边缘处的深度突变 / Spike noise: sudden depth jumps at edges
    - 边缘腐蚀：扩大物体周围的空洞 / Edge erosion: enlarges holes around objects
    - 深度不连续检测：大跳变处增加无效概率 / Discontinuity check: higher invalid probability at large jumps
    """

    def __init__(self, config: Optional[D435NoiseConfigV3] = None):
        self.config = config or D435NoiseConfigV3()

    def simulate(
        self,
        depth_m: ArrayLike,
        max_depth: float,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[ArrayLike, ArrayLike, Dict[str, float]]:
        """返回 (加噪深度, 无效掩码, 统计信息)。 / Return (noisy_depth, invalid_mask, stats)."""

        if rng is None:
            rng = np.random.default_rng(seed)
        elif seed is not None:
            rng = np.random.default_rng(seed)

        depth = depth_m.astype(np.float32)
        h, w = depth.shape
        cfg = self.config

        # --------------------------------------------------
        # ⓪ 左侧无效带（IR/RGB 视场差异） / Left invalid band (IR/RGB FOV difference)
        # --------------------------------------------------
        band_cols = 0
        if cfg.left_band_enabled:
            band_cols = cfg.left_band_px
            depth[:, :band_cols] = 0.0

        valid = depth > 0
        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = np.clip(depth[valid] / max_depth, 0.0, 1.0)

        # --------------------------------------------------
        # ① Ahn 2019 轴向噪声 / Ahn 2019 axial noise: σ(z) = c0 + c1*z + c2*z²
        # --------------------------------------------------
        z_clipped = np.clip(depth, 0, max_depth)
        sigma_axial = (cfg.axial_c0 + cfg.axial_c1 * z_clipped + cfg.axial_c2 * z_clipped ** 2)
        sigma_axial = sigma_axial * cfg.axial_scale
        axial_noise = rng.normal(0.0, 1.0, size=depth.shape) * sigma_axial * valid

        # --------------------------------------------------
        # ② Speckle 低频空间相关噪声 / Low-frequency, spatially correlated speckle noise
        # --------------------------------------------------
        speckle_field = _generate_low_frequency_noise(depth.shape, block=cfg.speckle_block, rng=rng)
        speckle_gain = (cfg.speckle_base + cfg.speckle_scale * depth_norm)
        speckle_noise = speckle_gain * speckle_field * depth

        # --------------------------------------------------
        # ③ 边缘强度（后续多处使用） / Edge strength (used by several steps below)
        # --------------------------------------------------
        edge_strength = _compute_edge_strength(depth, valid)

        # --------------------------------------------------
        # ④ 边缘系统偏移（飞点） / Systematic edge bias (flying pixels)
        # --------------------------------------------------
        edge_bias_scale = rng.uniform(cfg.edge_bias_min, cfg.edge_bias_max)
        edge_bias = edge_strength * edge_bias_scale * max_depth

        # --------------------------------------------------
        # ⑤ 【V3 新增】尖刺噪声 / [V3] Spiky artifacts
        # --------------------------------------------------
        spike_noise = np.zeros_like(depth)
        spike_count = 0
        if cfg.spike_enabled:
            # 在边缘强的区域随机产生尖刺 / Randomly place spikes in strong-edge regions
            spike_mask = (edge_strength > cfg.spike_edge_threshold) & valid
            spike_prob = rng.random(depth.shape)
            spike_locations = spike_mask & (spike_prob < cfg.spike_probability)

            if np.any(spike_locations):
                # 随机深度偏移（向前或向后） / Random depth shift (towards or away from the camera)
                spike_shift = rng.uniform(cfg.spike_depth_shift_min, cfg.spike_depth_shift_max, size=depth.shape)
                spike_direction = rng.choice([-1, 1], size=depth.shape)  # 随机方向 / random direction
                spike_noise[spike_locations] = depth[spike_locations] * spike_shift[spike_locations] * spike_direction[spike_locations]

                # 横向扩展尖刺宽度 / Widen the spikes horizontally
                if cfg.spike_width > 1:
                    kernel = np.ones((1, cfg.spike_width), dtype=np.uint8)
                    spike_locations_dilated = cv2.dilate(spike_locations.astype(np.uint8), kernel, iterations=1).astype(bool)
                    # 扩展区域施加衰减后的尖刺 / Apply attenuated spikes in the widened region
                    extended_region = spike_locations_dilated & ~spike_locations
                    if np.any(spike_locations):
                        mean_spike = np.mean(np.abs(spike_noise[spike_locations]))
                        spike_noise[extended_region] = mean_spike * 0.5 * spike_direction[extended_region]  # 衰减 50% / attenuated by 50%

                spike_count = np.sum(spike_locations)

        # --------------------------------------------------
        # ⑥ 空洞（边缘项 + 距离项，同 V2） / Holes (edge term + range term, as in V2)
        # --------------------------------------------------
        hole_prob = (cfg.hole_base
                     + cfg.hole_edge_scale * edge_strength
                     + cfg.hole_range_scale * depth_norm)
        hole_prob = hole_prob * np.exp(-cfg.hole_depth_decay * depth_norm)
        hole_prob = np.clip(hole_prob, 0.0, 0.5)

        low_freq_cluster = _generate_low_frequency_noise((h, w), block=cfg.hole_cluster_block, rng=rng)
        hole_mask = (rng.random((h, w)) < hole_prob) & (low_freq_cluster > cfg.hole_cluster_threshold)

        # --------------------------------------------------
        # ⑦ 【V3 新增】边缘腐蚀（成片扩大边界空洞） / [V3] Edge erosion (contiguous holes along boundaries)
        # --------------------------------------------------
        edge_erosion_mask = np.zeros_like(hole_mask)
        erosion_count = 0
        if cfg.edge_erosion_enabled:
            # 检测强边缘区域 / Detect strong-edge regions
            strong_edges = edge_strength > cfg.edge_erosion_threshold

            # 形态学膨胀，扩大可能失效的区域 / Morphological dilation enlarges the potentially invalid region
            kernel = np.ones((cfg.edge_erosion_kernel, cfg.edge_erosion_kernel), dtype=np.uint8)
            edges_dilated = cv2.dilate(strong_edges.astype(np.uint8), kernel, iterations=2).astype(bool)

            # 用块更大的低频噪声场控制边缘空洞成片分布（而非逐像素随机）
            # A low-frequency field with larger blocks makes edge holes contiguous (rather than per-pixel random)
            edge_cluster_field = _generate_low_frequency_noise((h, w), block=8, rng=rng)

            # 边缘区域内，由低频场和基础概率共同决定是否失效
            # Within the edge region, invalidity is decided by the low-frequency field and a base probability
            edge_region = edges_dilated & ~hole_mask  # 不重复已有空洞 / skip existing holes
            edge_base_prob = cfg.edge_erosion_prob * 0.7  # 降低基础概率 / reduced base probability
            edge_cluster_threshold = 0.3  # 较低阈值让空洞更容易连片 / a lower threshold makes holes merge more easily

            edge_erosion_mask = (edge_region
                                & (edge_cluster_field > edge_cluster_threshold)
                                & (rng.random(depth.shape) < edge_base_prob))

            # 形态学闭运算，填补内部小缝隙，让空洞更连续 / Morphological closing fills small gaps so holes are more contiguous
            if np.any(edge_erosion_mask):
                closing_kernel = np.ones((3, 3), dtype=np.uint8)
                edge_erosion_mask = cv2.morphologyEx(
                    edge_erosion_mask.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    closing_kernel,
                    iterations=1
                ).astype(bool)

            erosion_count = np.sum(edge_erosion_mask)

        # --------------------------------------------------
        # ⑧ 【V3 新增】深度不连续检测 / [V3] Depth-discontinuity check
        # --------------------------------------------------
        discontinuity_mask = np.zeros_like(hole_mask)
        discontinuity_count = 0
        if cfg.discontinuity_enabled:
            # 计算水平和垂直方向的深度梯度 / Horizontal and vertical depth gradients
            depth_filled = depth.copy()
            depth_filled[~valid] = 0.0

            gx = np.abs(np.diff(depth_filled, axis=1, prepend=0))
            gy = np.abs(np.diff(depth_filled, axis=0, prepend=0))
            depth_jump = np.maximum(gx, gy)

            # 大跳变处增加无效概率 / Higher invalid probability at large jumps
            jump_locations = (depth_jump > cfg.discontinuity_threshold) & valid
            discontinuity_prob = rng.random(depth.shape)
            discontinuity_mask = jump_locations & (discontinuity_prob < cfg.discontinuity_invalidate_prob)
            discontinuity_count = np.sum(discontinuity_mask)

        # --------------------------------------------------
        # ⑨ 随机 dropout（随距离衰减，同 V2） / Random dropout (decays with range, as in V2)
        # --------------------------------------------------
        dropout_prob = cfg.dropout_base * np.exp(-cfg.dropout_depth_decay * depth_norm)
        dropout_mask = rng.random(depth.shape) < dropout_prob

        # --------------------------------------------------
        # ⑩ 叠加所有噪声 / Combine all noise terms
        # --------------------------------------------------
        noisy = depth + axial_noise + speckle_noise + edge_bias + spike_noise
        noisy = np.clip(noisy, 0.0, max_depth)

        # 合并所有无效掩码（V2 + V3） / Merge all invalid masks (V2 + V3)
        invalid_mask = hole_mask | dropout_mask | edge_erosion_mask | discontinuity_mask

        # 填充无效值 / Fill invalid pixels
        if cfg.invalid_fill_mode == "maxdepth":
            noisy[invalid_mask] = max_depth
        else:
            noisy[invalid_mask] = 0.0

        # 确保左侧无效带包含在最终掩码中 / Make sure the left band is in the final mask
        if cfg.left_band_enabled:
            invalid_mask[:, :band_cols] = True

        # --------------------------------------------------
        # 统计信息（V2 + V3） / Statistics (V2 + V3)
        # --------------------------------------------------
        stats = {
            "axial_mm_std": float(np.mean(sigma_axial[valid]) * 1000.0) if np.any(valid) else 0.0,
            "speckle_mm_std": float(np.sqrt(np.mean(np.square(speckle_noise[valid]))) * 1000.0)
            if np.any(valid) else 0.0,
            "hole_ratio": float(np.mean(hole_mask)),
            "dropout_ratio": float(np.mean(dropout_mask)),
            "edge_erosion_ratio": float(np.mean(edge_erosion_mask)),
            "discontinuity_ratio": float(np.mean(discontinuity_mask)),
            "invalid_ratio": float(np.mean(invalid_mask)),
            "edge_bias_cm": float(np.mean(edge_bias[edge_strength > 0.3]) * 100.0)
            if np.any(edge_strength > 0.3) else 0.0,
            "spike_count": int(spike_count),
            "erosion_count": int(erosion_count),
            "discontinuity_count": int(discontinuity_count),
            "left_band_px": band_cols if cfg.left_band_enabled else 0,
        }

        return noisy.astype(np.float32), invalid_mask, stats


__all__ = [
    "D435NoiseConfig", "D435DepthNoiseSimulator",
    "D435NoiseConfigV2", "D435DepthNoiseSimulatorV2",
    "D435NoiseConfigV3", "D435DepthNoiseSimulatorV3"
]
