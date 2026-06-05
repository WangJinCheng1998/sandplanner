#!/usr/bin/env python3
"""
B-spline 控制点预测的 DDPM 训练脚本。
DDPM training script for B-spline control point prediction.

使用 ResNet18 提取深度图像特征，Transformer Encoder 处理 token 序列，DDPM 生成 B-spline 控制点；
当前使用官方 UNet1DConditionModel 实现。
Extracts depth image features with ResNet18, processes the token sequence with a Transformer
Encoder, and generates B-spline control points with a DDPM. Now built on the official
UNet1DConditionModel implementation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from diffusers import DDPMScheduler
import timm
# 将 SanD-planner 根目录加入 sys.path，以便导入 sand_planner 包
# Add the SanD-planner root to sys.path so the sand_planner package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sand_planner.nn.models.unet.unet_1d_condition import UNet1DConditionModel
from diffusers import DPMSolverMultistepScheduler
from sand_planner.training.dataloader_bspline import create_bspline_dataloader
from sand_planner.utils.normalize import TrajectoryNormalizer

class DepthImageEncoder(nn.Module):
    """深度图像编码器，使用 ResNet18 提取空间特征图。 / Depth image encoder that extracts spatial feature maps with ResNet18."""

    def __init__(self, feature_dim: int = 512, fusion_strategy: str = 'concat', use_tea: bool = False):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.use_tea = use_tea

        # 使用 ResNet18 提取空间特征，取 stride=16 的特征层
        # Use ResNet18 to extract spatial features, taking the stride=16 feature stage
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=False,
            features_only=True,
            in_chans=1,
            # 改用第 3 层（stride=16）而非 stride=32 / Use stage 3 (stride=16) instead of stride=32
            out_indices=[3],
        )

        # 获取 backbone 输出维度与空间尺寸 / Probe the backbone output dimension and spatial size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 168, 224)
            features = self.backbone(dummy_input)
            backbone_out_dim = features[0].shape[1]  # 通道数 / number of channels
            original_h = features[0].shape[2]        # 原始空间高度 / original spatial height
            original_w = features[0].shape[3]        # 原始空间宽度 / original spatial width

        # 使用 AdaptiveAvgPool2d 池化到固定网格 8x12=96 tokens/帧
        # Use AdaptiveAvgPool2d to pool to a fixed 8x12=96 tokens/frame grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 12))  # 固定输出 8x12=96 个 tokens / fixed 8x12=96 tokens output
        self.spatial_h = 8
        self.spatial_w = 12
        self.num_spatial_tokens = self.spatial_h * self.spatial_w  # 96 个 tokens / 96 tokens
        
        print(f"🔧 空间特征图: {original_h}×{original_w} → 池化到{self.spatial_h}×{self.spatial_w}={self.num_spatial_tokens}tokens")
        print(f"📊 每帧tokens: {original_h * original_w} → {self.num_spatial_tokens}")
        print(f"🔄 多帧融合策略: {self.fusion_strategy}")
        
        # 1x1 卷积投影到目标维度 / 1x1 convolution projecting to the target dimension
        self.spatial_proj = nn.Conv2d(backbone_out_dim, feature_dim, kernel_size=1)

        # 时序注意力机制（仅在 attention 模式下使用）
        # Temporal attention (only used in the attention fusion mode)
        if self.fusion_strategy == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )

        # 2D 正弦位置编码，注册为 buffer，以更好地表示空间位置关系
        # 2D sinusoidal positional encoding, registered as a buffer to better encode spatial layout
        pos_embed_2d = self._create_2d_pos_encoding(feature_dim, self.spatial_h, self.spatial_w)
        self.register_buffer('pos_embed_2d', pos_embed_2d)

        # 时间步嵌入（用于多帧序列）/ Timestep embedding (for multi-frame sequences)
        self.time_embed = nn.Embedding(32, feature_dim)  # 支持最多 32 帧 / supports up to 32 frames

        self.feature_dim = feature_dim

        # TEA-lite: 运动编码器，用于捕捉帧间差分
        # TEA-lite: motion encoder used to capture inter-frame differences
        if self.use_tea:
            # Motion encoder 输入: [当前帧, 差分] = 2*feature_dim
            # Motion encoder input: [current frame, difference] = 2*feature_dim
            self.motion_encoder = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),  # 输入是 2*C / input is 2*C
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            )
            # 关键修复：对 motion encoder 做小初始化，避免训练初期产生过大扰动
            # Critical fix: small init for the motion encoder to avoid large perturbations early in training
            for module in self.motion_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print(f"🔥 TEA-lite 已启用: 将生成 3 个运动差分 token (输入=[g_t, Δg_t], 小初始化)")
        
    def _create_2d_pos_encoding(self, d_model: int, height: int, width: int) -> torch.Tensor:
        """创建 2D 正弦位置编码。 / Create a 2D sinusoidal positional encoding."""
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D positional encoding"

        # 创建位置编码矩阵 / Allocate the positional encoding matrix
        pos_encoding = torch.zeros(height, width, d_model)

        # 计算频率项 / Compute the frequency term
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))

        # 为每个位置生成编码 / Generate the encoding for each position
        for h in range(height):
            for w in range(width):
                # H 方向编码（使用偶数索引）/ Height-axis encoding (even indices)
                pos_encoding[h, w, 0::4] = torch.sin(h * div_term)
                pos_encoding[h, w, 1::4] = torch.cos(h * div_term)

                # W 方向编码（使用奇数索引）/ Width-axis encoding (odd indices)
                pos_encoding[h, w, 2::4] = torch.sin(w * div_term)
                pos_encoding[h, w, 3::4] = torch.cos(w * div_term)

        return pos_encoding.view(1, height * width, d_model)  # (1, H*W, d_model)

    def _create_1d_pos_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """创建 1D 正弦位置编码。 / Create a 1D sinusoidal positional encoding."""
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # (1, max_len, d_model)
        
    def forward(self, depth_images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播。 / Forward pass.

        Args:
            depth_images: (batch_size, seq_len, 1, H, W) 深度图像序列 / depth image sequence.
        Returns:
            features: (batch_size, seq_len * num_spatial_tokens, feature_dim) 空间特征 token / spatial feature tokens.
            motion_tokens: 启用 use_tea 时为 (batch_size, 3, feature_dim) 的 3 个帧间差分 token，否则为 None /
                3 inter-frame difference tokens of shape (batch_size, 3, feature_dim) when use_tea is set, else None.
        """
        batch_size, seq_len, channels, height, width = depth_images.shape

        # 重塑为 (batch_size * seq_len, channels, H, W) / Reshape to (batch_size * seq_len, channels, H, W)
        depth_images = depth_images.reshape(batch_size * seq_len, channels, height, width)

        # 通过 ResNet 提取空间特征图 / Extract spatial feature maps with the ResNet backbone
        features = self.backbone(depth_images)[0]  # (batch_size * seq_len, C, H', W')

        # 1x1 卷积投影维度 / Project channels with the 1x1 convolution
        features = self.spatial_proj(features)  # (batch_size * seq_len, feature_dim, H', W')

        # 应用自适应池化到固定网格 8x12=96 tokens / Adaptive-pool to the fixed 8x12=96 token grid
        features = self.adaptive_pool(features)  # (batch_size * seq_len, feature_dim, 8, 12)

        # 展平空间维度为 tokens / Flatten the spatial dims into tokens
        spatial_features = features.flatten(2).transpose(1, 2)  # (batch_size * seq_len, 96, feature_dim)

        # 添加 2D 位置编码（buffer 会自动跟随模型设备）
        # Add the 2D positional encoding (the buffer follows the model device automatically)
        spatial_features = spatial_features + self.pos_embed_2d.to(spatial_features.dtype)

        # 重塑回序列维度 / Reshape back to include the sequence dimension
        spatial_features = spatial_features.reshape(
            batch_size, seq_len, self.num_spatial_tokens, self.feature_dim
        )  # (batch_size, seq_len, num_spatial_tokens, feature_dim)

        # TEA-lite: 在添加时间步嵌入之前提取运动信息（关键修复）
        # TEA-lite: extract motion information before adding the timestep embedding (critical fix)
        motion_tokens = None
        if self.use_tea and seq_len > 1:
            # 1. 提取每帧的全局特征向量（空间平均池化）；关键在于使用未加时间步嵌入的纯视觉特征
            # 1. Extract each frame's global feature via spatial average pooling; crucially on the pure
            #    visual features before the timestep embedding is added
            global_feat_raw = spatial_features.mean(dim=2)  # (B, seq_len, feature_dim)

            # 2. 计算帧间差分: 4 帧 -> 3 个差分向量 / Compute inter-frame differences: 4 frames -> 3 difference vectors
            # delta[0] = frame[1] - frame[0]  (第 1->2 帧的运动 / motion frame 1->2)
            # delta[1] = frame[2] - frame[1]  (第 2->3 帧的运动 / motion frame 2->3)
            # delta[2] = frame[3] - frame[2]  (第 3->4 帧的运动 / motion frame 3->4)
            delta_feat = global_feat_raw[:, 1:] - global_feat_raw[:, :-1]  # (B, seq_len-1, feature_dim)

            # 3. Motion encoder 输入: 拼接 [当前帧特征, 帧间差分]，既保留绝对位置信息 (g_t)，又保留相对运动信息 (Δg_t)
            # 3. Motion encoder input: concat [current-frame feature, inter-frame difference], keeping both the
            #    absolute position info (g_t) and the relative motion info (Δg_t)
            current_frames = global_feat_raw[:, 1:]  # (B, seq_len-1, feature_dim) - 取后 3 帧 / last 3 frames
            motion_input = torch.cat([current_frames, delta_feat], dim=-1)  # (B, 3, 2*feature_dim)

            # 4. 通过 MLP 编码为运动 token / Encode into motion tokens via the MLP
            B, num_deltas, C2 = motion_input.shape  # C2 = 2*feature_dim
            motion_flat = motion_input.reshape(B * num_deltas, C2)  # (B*3, 2*C)
            motion_encoded = self.motion_encoder(motion_flat)  # (B*3, C)
            motion_tokens = motion_encoded.reshape(B, num_deltas, self.feature_dim)  # (B, 3, C)

        # 添加时间步嵌入（在运动提取之后）/ Add the timestep embedding (after motion extraction)
        time_ids = torch.arange(seq_len, device=spatial_features.device)  # (seq_len,)
        time_embeds = self.time_embed(time_ids).to(spatial_features.dtype)  # (seq_len, feature_dim)
        time_embeds = time_embeds.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, feature_dim)
        spatial_features = spatial_features + time_embeds

        # 展平为统一的 token 序列 / Flatten into a unified token sequence
        if self.fusion_strategy == 'concat' or seq_len == 1:
            # 原始拼接策略或单帧 / Original concat strategy or single frame
            output_features = spatial_features.reshape(
                batch_size, seq_len * self.num_spatial_tokens, self.feature_dim
            )  # (batch_size, seq_len * num_spatial_tokens, feature_dim)
        elif self.fusion_strategy == 'average':
            # 时序平均融合：将多帧平均到单帧的 token 数量
            # Temporal average fusion: average multiple frames down to a single frame's token count
            output_features = spatial_features.mean(dim=1)  # (batch_size, num_spatial_tokens, feature_dim)
        elif self.fusion_strategy == 'attention':
            # 时序注意力融合 / Temporal attention fusion
            # 重塑为 (batch_size * num_spatial_tokens, seq_len, feature_dim)
            # Reshape to (batch_size * num_spatial_tokens, seq_len, feature_dim)
            spatial_for_attn = spatial_features.transpose(1, 2).reshape(
                batch_size * self.num_spatial_tokens, seq_len, self.feature_dim
            )

            # 应用注意力：在每个空间位置上融合时序信息
            # Apply attention: fuse temporal information at each spatial location
            attn_output, _ = self.temporal_attention(
                spatial_for_attn, spatial_for_attn, spatial_for_attn
            )  # (batch_size * num_spatial_tokens, seq_len, feature_dim)

            # 时序维度平均池化 / Average-pool over the temporal dimension
            attn_pooled = attn_output.mean(dim=1)  # (batch_size * num_spatial_tokens, feature_dim)

            # 重塑回 (batch_size, num_spatial_tokens, feature_dim)
            # Reshape back to (batch_size, num_spatial_tokens, feature_dim)
            output_features = attn_pooled.reshape(
                batch_size, self.num_spatial_tokens, self.feature_dim
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return output_features, motion_tokens

class TrajectoryConditionEncoder(nn.Module):
    """轨迹条件编码器。 / Trajectory condition encoder."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, feature_dim: int = 512):
        super().__init__()
        # 输入是 3 维：(x, y, z) / Input is 3-dimensional: (x, y, z)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.feature_dim = feature_dim
        
    def forward(self, end_relative_pose: torch.Tensor) -> torch.Tensor:
        """前向传播。 / Forward pass.

        Args:
            end_relative_pose: (batch_size, 3) 终点相对坐标 / endpoint relative coordinates.
        Returns:
            encoded: (batch_size, feature_dim) 编码后的特征 / encoded features.
        """
        return self.encoder(end_relative_pose)


class InitialTurnEncoder(nn.Module):
    """初始转向编码器，编码轨迹起始的转弯方向（1 维 Y 分量）。 / Initial-turn encoder for the trajectory's starting turn direction (1-D Y component).

    用于提供历史运动信息，帮助模型生成更平滑连续的轨迹；输入是归一化后的 (CP[1] + CP[2]) / 2 的 Y 分量。
    Provides historical motion cues to help the model generate smoother, more continuous trajectories;
    the input is the normalized Y component of (CP[1] + CP[2]) / 2.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # 简单 MLP: 1 维 -> feature_dim / Simple MLP: 1-D -> feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, feature_dim)
        )

        # Null token: 表示"没有历史转向信息"（推理时首次规划）
        # Null token: represents "no historical turn information" (the first plan during inference)
        self.null_token = nn.Parameter(torch.zeros(feature_dim))

        # 小初始化以避免破坏已有特征 / Small init to avoid disrupting existing features
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)
        
        print(f"🔄 InitialTurnEncoder 已创建: 1维转向 → {feature_dim}维特征")
    
    def forward(self, initial_turn: torch.Tensor, has_initial_turn: torch.Tensor) -> torch.Tensor:
        """前向传播。 / Forward pass.

        Args:
            initial_turn: (B,) 或 (B, 1) 初始转向值（归一化后的 Y 分量）/ initial turn value, shape (B,) or (B, 1) (normalized Y component).
            has_initial_turn: (B,) bool，是否有有效的初始转向 / bool mask of whether a valid initial turn exists.
        Returns:
            features: (B, feature_dim) 编码后的特征 / encoded features.
        """
        # 确保输入是 (B, 1) 形状 / Ensure the input has shape (B, 1)
        if initial_turn.dim() == 1:
            initial_turn = initial_turn.unsqueeze(1)  # (B,) -> (B, 1)

        batch_size = initial_turn.shape[0]
        device = initial_turn.device

        # 编码转向 / Encode the turn
        encoded = self.encoder(initial_turn)  # (B, feature_dim)

        # 对于没有历史的样本，使用 null token / For samples without history, use the null token
        null_expanded = self.null_token.unsqueeze(0).expand(batch_size, -1)  # (B, feature_dim)

        # 确保 has_initial_turn 是 tensor / Ensure has_initial_turn is a tensor
        if isinstance(has_initial_turn, bool):
            has_mask = torch.full((batch_size,), has_initial_turn, dtype=torch.float32, device=device)
        else:
            has_mask = has_initial_turn.float()
        
        has_mask = has_mask.unsqueeze(1)  # (B, 1)

        output = encoded * has_mask + null_expanded * (1 - has_mask)
        return output  # (B, feature_dim)


class ConcatConditionEncoder(nn.Module):
    """条件编码器，拼接深度与轨迹特征后通过 Transformer 处理。 / Condition encoder that concatenates depth and trajectory features and processes them with a Transformer."""

    def __init__(self,
                 feature_dim: int = 256,
                 seq_len: int = 2,
                 num_transformer_layers: int = 4,
                 num_heads: int = 4,
                 cfg_drop_depth: bool = False,
                 cfg_drop_trajectory: bool = True,
                 fusion_strategy: str = 'concat',
                 use_tea: bool = False,
                 use_type_embed: bool = True,
                 use_initial_turn: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.cfg_drop_depth = cfg_drop_depth  # 是否在 CFG 时丢弃深度条件 / whether to drop the depth condition under CFG
        self.cfg_drop_trajectory = cfg_drop_trajectory  # 是否在 CFG 时丢弃轨迹条件 / whether to drop the trajectory condition under CFG
        self.fusion_strategy = fusion_strategy
        self.use_tea = use_tea  # 是否使用 TEA 运动 token / whether to use TEA motion tokens
        self.use_type_embed = use_type_embed  # 是否使用类型编码 / whether to use type embedding
        self.use_initial_turn = use_initial_turn  # 是否使用初始转向 / whether to use the initial turn

        # 深度图像编码器 / Depth image encoder
        self.depth_encoder = DepthImageEncoder(feature_dim, fusion_strategy=fusion_strategy, use_tea=use_tea)

        # 轨迹条件编码器 / Trajectory condition encoder
        self.trajectory_encoder = TrajectoryConditionEncoder(
            input_dim=3, feature_dim=feature_dim  # 输入 3 维：(x, y, z) / 3-D input: (x, y, z)
        )

        # 初始转向编码器 / Initial-turn encoder
        if self.use_initial_turn:
            self.initial_turn_encoder = InitialTurnEncoder(feature_dim)
            self.turn_ln = nn.LayerNorm(feature_dim)

        # LayerNorm
        self.depth_ln = nn.LayerNorm(feature_dim)
        self.traj_ln = nn.LayerNorm(feature_dim)

        # 运动 token 的 LayerNorm / LayerNorm for the motion tokens
        if self.use_tea:
            self.motion_ln = nn.LayerNorm(feature_dim)

        # 类型编码（Type Embedding），用于区分不同模态的 token / Type embedding to distinguish token modalities
        # 类型 0: depth tokens（视觉空间特征）/ type 0: depth tokens (visual spatial features)
        # 类型 1: motion tokens（运动差分特征）/ type 1: motion tokens (motion-difference features)
        # 类型 2: initial_turn token（初始转向特征）/ type 2: initial_turn token (initial-turn feature)
        # 类型 3: trajectory tokens（目标轨迹特征）/ type 3: trajectory tokens (target trajectory features)
        if self.use_type_embed:
            num_types = 4 if self.use_initial_turn else 3
            self.type_embed = nn.Embedding(num_types, feature_dim)
            # 关键修复：小初始化以避免破坏已有特征分布 / Critical fix: small init to avoid disrupting existing feature distribution
            nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)
            type_names = "depth/motion/turn/trajectory" if self.use_initial_turn else "depth/motion/trajectory"
            print(f"🎯 Type Embedding 已启用: 区分 {type_names} {num_types}种模态 (小初始化 std=0.02)")

        # 最简单的 Transformer Encoder / The simplest Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 总位移预测头：预测第 1 帧到第 4 帧的相对位姿变化
        # Total-displacement prediction head: predicts the relative pose change from frame 1 to frame 4
        self.sequence_pose_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            # [dx, dy, dz, dyaw, dpitch] - 总位移（完整 5 维）/ total displacement (full 5-D)
            nn.Linear(128, 5),
            nn.Tanh()
        )

        # 位姿约束范围 / Pose constraint ranges
        self.max_translation = 1.0  # 4 帧之间最大位移 1 米 / max translation of 1 m across 4 frames
        self.max_rotation = 0.785   # 最大旋转 45 度 / max rotation of 45 degrees

    def forward(self,
                depth_sequence: torch.Tensor,
                end_relative_pose: torch.Tensor,
                initial_turn: Optional[torch.Tensor] = None,
                has_initial_turn: Optional[torch.Tensor] = None,
                cfg_mask: Optional[torch.Tensor] = None,
                return_pose_features: bool = False) -> torch.Tensor:
        """将深度序列空间 tokens 与目标姿态分别编码、LayerNorm 后拼接，再通过 GPT 式 Transformer 处理。 / Encode the depth-sequence spatial tokens and the target pose separately, LayerNorm and concatenate them, then process with a GPT-style Transformer.

        Args:
            depth_sequence: (B, seq_len, 1, H, W) 深度图像序列 / depth image sequence.
            end_relative_pose: (B, 3) 轨迹终点相对坐标 / trajectory endpoint relative coordinates.
            initial_turn: (B,) 初始转向值（可选）/ initial turn value (optional).
            has_initial_turn: (B,) 是否有有效的初始转向（可选）/ whether a valid initial turn exists (optional).
            cfg_mask: (B,) CFG 掩码，True 表示使用条件、False 表示丢弃条件（用于 CFG 训练）/ CFG mask, True keeps the condition and False drops it (for CFG training).
            return_pose_features: bool，是否返回用于位姿预测的中间特征 / whether to also return the intermediate features used for pose prediction.
        """

        # 编码深度图像序列为空间 tokens 与运动 tokens / Encode the depth image sequence into spatial tokens and motion tokens
        depth_tokens, motion_tokens = self.depth_encoder(depth_sequence)
        # depth_tokens: (B, seq_len*N, C) 或 (B, N, C)，取决于 fusion_strategy / shape (B, seq_len*N, C) or (B, N, C) depending on fusion_strategy
        # motion_tokens: 启用 use_tea 时为 (B, 3, C)，否则为 None / (B, 3, C) when use_tea is set, else None

        # 编码轨迹终点 / Encode the trajectory endpoint
        traj_features = self.trajectory_encoder(end_relative_pose).unsqueeze(1)  # (B, 1, C)

        # 编码初始转向 / Encode the initial turn
        turn_token = None
        if self.use_initial_turn:
            batch_size_local = depth_tokens.shape[0]
            device_local = depth_tokens.device

            if initial_turn is not None:
                # 有 initial_turn 输入，使用编码器处理 / initial_turn provided, process it through the encoder
                turn_features = self.initial_turn_encoder(initial_turn, has_initial_turn)  # (B, C)
            else:
                # 没有 initial_turn 输入（推理时冷启动），使用 null token / no initial_turn (cold start at inference), use the null token
                turn_features = self.initial_turn_encoder.null_token.unsqueeze(0).expand(batch_size_local, -1)  # (B, C)

            turn_token = self.turn_ln(turn_features).unsqueeze(1)  # (B, 1, C)

        # 先进行 LayerNorm，再应用 CFG mask / Apply LayerNorm first, then the CFG mask
        enhanced_depth_tokens = self.depth_ln(depth_tokens)
        enhanced_traj_features = self.traj_ln(traj_features)  # 先 LayerNorm / LayerNorm first

        # TEA 运动 tokens 的 LayerNorm / LayerNorm for the TEA motion tokens
        if motion_tokens is not None:
            enhanced_motion_tokens = self.motion_ln(motion_tokens)  # (B, 3, C)
        else:
            enhanced_motion_tokens = None

        # 复制目标 token 3 次以增加权重并兼容多模态架构
        # Replicate the target token 3 times to increase its weight and fit the multimodal architecture
        enhanced_traj_features = enhanced_traj_features.repeat(1, 3, 1)  # (B, 1, C) -> (B, 3, C)

        # 添加类型编码（Type Embedding）/ Add the type embedding
        if self.use_type_embed:
            batch_size = enhanced_depth_tokens.shape[0]
            device = enhanced_depth_tokens.device

            # 类型 0: depth tokens / type 0: depth tokens
            num_depth = enhanced_depth_tokens.shape[1]
            type_ids_depth = torch.zeros(batch_size, num_depth, dtype=torch.long, device=device)
            type_embed_depth = self.type_embed(type_ids_depth)  # (B, num_depth, C)
            enhanced_depth_tokens = enhanced_depth_tokens + type_embed_depth

            # 类型 1: motion tokens（如果存在）/ type 1: motion tokens (if present)
            if enhanced_motion_tokens is not None:
                type_ids_motion = torch.ones(batch_size, 3, dtype=torch.long, device=device)
                type_embed_motion = self.type_embed(type_ids_motion)  # (B, 3, C)
                enhanced_motion_tokens = enhanced_motion_tokens + type_embed_motion

            # 类型 2: initial_turn token（如果存在）/ type 2: initial_turn token (if present)
            if turn_token is not None:
                type_ids_turn = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
                type_embed_turn = self.type_embed(type_ids_turn)  # (B, 1, C)
                turn_token = turn_token + type_embed_turn

            # 类型 3（或 2）: trajectory tokens / type 3 (or 2): trajectory tokens
            traj_type_id = 3 if self.use_initial_turn else 2
            type_ids_traj = torch.full((batch_size, 3), traj_type_id, dtype=torch.long, device=device)
            type_embed_traj = self.type_embed(type_ids_traj)  # (B, 3, C)
            enhanced_traj_features = enhanced_traj_features + type_embed_traj

        # CFG 处理：在 LayerNorm 之后用 mask 置零，避免 LayerNorm 泄漏偏置
        # CFG handling: zero out via the mask after LayerNorm to avoid leaking the LayerNorm bias
        if cfg_mask is not None:
            # cfg_mask 为 True 的样本使用正常条件，False 的样本置零 / samples with cfg_mask=True keep the condition, others are zeroed
            batch_size = enhanced_traj_features.shape[0]
            mask_expanded = cfg_mask.view(batch_size, 1, 1)  # (B, 1, 1)

            # 轨迹条件：根据配置决定是否丢弃 / Trajectory condition: drop it depending on the config
            if self.cfg_drop_trajectory:
                enhanced_traj_features = enhanced_traj_features * mask_expanded.float()

            # 深度条件：根据配置决定是否丢弃 / Depth condition: drop it depending on the config
            if self.cfg_drop_depth:
                enhanced_depth_tokens = enhanced_depth_tokens * mask_expanded.float()
                # 如果丢弃深度，也丢弃运动信息（因为运动是从深度序列提取的）
                # If depth is dropped, also drop the motion info (motion is extracted from the depth sequence)
                if enhanced_motion_tokens is not None:
                    enhanced_motion_tokens = enhanced_motion_tokens * mask_expanded.float()

        # 拼接经过 LayerNorm 与 CFG 处理的 tokens: [depth, motion, turn, trajectory]
        # Concatenate the LayerNorm-ed and CFG-processed tokens: [depth, motion, turn, trajectory]
        tokens_to_concat = [enhanced_depth_tokens]
        if enhanced_motion_tokens is not None:
            tokens_to_concat.append(enhanced_motion_tokens)  # 添加 3 个运动 token / add the 3 motion tokens
        if turn_token is not None:
            tokens_to_concat.append(turn_token)  # 添加 1 个转向 token / add the 1 turn token
        tokens_to_concat.append(enhanced_traj_features)

        combined_tokens = torch.cat(tokens_to_concat, dim=1)
        # 完整版形状: (B, depth_tokens+3(motion)+1(turn)+3(traj), C) / full shape: (B, depth_tokens+3(motion)+1(turn)+3(traj), C)

        # 通过简单的 Transformer Encoder 处理拼接后的 tokens / Process the concatenated tokens with the simple Transformer Encoder
        transformer_output = self.transformer(combined_tokens)

        # CFG 后处理：Transformer 的 self-attention 会破坏零 token 状态，需要重新应用 mask
        # CFG post-processing: the Transformer self-attention contaminates the zeroed tokens, so re-apply the mask
        if cfg_mask is not None:
            batch_size = transformer_output.shape[0]
            mask_expanded = cfg_mask.view(batch_size, 1, 1)  # (B, 1, 1)

            # 计算轨迹 token 的位置（最后 3 个）/ Locate the trajectory tokens (the last 3)
            traj_start_idx = -3

            # 重新对轨迹 token 应用 mask / Re-apply the mask to the trajectory tokens
            if self.cfg_drop_trajectory:
                transformer_output[:, traj_start_idx:, :] = transformer_output[:, traj_start_idx:, :] * mask_expanded.float()

            # 如果也丢弃深度条件 / If the depth condition is also dropped
            if self.cfg_drop_depth:
                # 深度 tokens + 运动 tokens 的范围 / Range covering the depth tokens and motion tokens
                depth_motion_end_idx = traj_start_idx
                transformer_output[:, :depth_motion_end_idx, :] = transformer_output[:, :depth_motion_end_idx, :] * mask_expanded.float()

        # 如果需要返回位姿预测特征 / If the pose-prediction features are requested
        if return_pose_features:
            # 直接使用 Transformer 输出的全局特征（所有 tokens 的平均），充分利用其处理后的信息，包括深度与轨迹的交互
            # Use the global feature of the Transformer output (mean over all tokens) to fully exploit its
            # processed information, including the depth-trajectory interaction
            pose_features = transformer_output.mean(dim=1)  # (B, C) - 全局平均池化 / global average pooling
            return transformer_output, pose_features

        return transformer_output
 

class CrossAttnConditionEncoder(nn.Module):
    """条件编码器，使用 Cross Attention 将轨迹目标注入到深度特征中。 / Condition encoder that injects the trajectory target into the depth features via cross attention."""

    def __init__(self,
                 feature_dim: int = 256,
                 seq_len: int = 2,
                 num_transformer_layers: int = 2,  # 默认改为 2 层，呼应之前的分析 / default lowered to 2 layers per prior analysis
                 num_heads: int = 4,
                 cfg_drop_depth: bool = False,
                 cfg_drop_trajectory: bool = True,
                 fusion_strategy: str = 'concat',
                 use_tea: bool = False,
                 use_type_embed: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.cfg_drop_depth = cfg_drop_depth
        self.cfg_drop_trajectory = cfg_drop_trajectory
        self.fusion_strategy = fusion_strategy
        self.use_tea = use_tea  # 是否使用 TEA 运动 token / whether to use TEA motion tokens
        self.use_type_embed = use_type_embed  # 是否使用类型编码 / whether to use type embedding

        # 深度图像编码器 / Depth image encoder
        self.depth_encoder = DepthImageEncoder(feature_dim, fusion_strategy=fusion_strategy, use_tea=use_tea)

        # 轨迹条件编码器 / Trajectory condition encoder
        self.trajectory_encoder = TrajectoryConditionEncoder(
            input_dim=3, feature_dim=feature_dim
        )

        # LayerNorm
        self.depth_ln = nn.LayerNorm(feature_dim)
        self.traj_ln = nn.LayerNorm(feature_dim)

        # 运动 token 的 LayerNorm / LayerNorm for the motion tokens
        if self.use_tea:
            self.motion_ln = nn.LayerNorm(feature_dim)

        # 类型编码（Type Embedding）/ Type embedding
        if self.use_type_embed:
            self.type_embed = nn.Embedding(3, feature_dim)
            print(f"🎯 Type Embedding 已启用 (CrossAttn): 区分 depth/motion/trajectory")

        # 混合架构：Self-Attention（理解环境）+ Cross-Attention（注入目标）
        # Hybrid architecture: self-attention (understand the scene) + cross-attention (inject the target)
        # Stage 1: Self-Attention - 视觉 token 之间的交互，理解障碍物空间结构
        # Stage 1: self-attention - interactions among visual tokens to understand obstacle spatial structure
        # 关键发现：3 层 Self-Attn 是感知低矮障碍物的最小深度
        # Key finding: 3 self-attention layers are the minimum depth needed to perceive low obstacles
        #   - 第 1 层：理解局部边缘和纹理 / layer 1: local edges and textures
        #   - 第 2 层：理解空间关系和形状 / layer 2: spatial relations and shapes
        #   - 第 3 层：理解地面 vs 障碍物的语义差异 / layer 3: ground-vs-obstacle semantic distinction
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        # 3+1 分割：75% 给视觉理解，25% 给目标注入 / 3+1 split: 75% for visual understanding, 25% for target injection
        num_self_layers = max(1, int(num_transformer_layers * 0.75))
        self.self_attention = nn.TransformerEncoder(self_attn_layer, num_layers=num_self_layers)

        # Stage 2: Cross-Attention - 将目标信息注入到已理解的视觉特征中
        # Stage 2: cross-attention - inject the target information into the already-understood visual features
        # 1 层 Cross-Attn 足够：视觉特征已充分编码，只需高效注入目标
        # 1 cross-attention layer is enough: the visual features are well encoded, only efficient target injection is needed
        cross_attn_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        num_cross_layers = max(1, num_transformer_layers - num_self_layers)
        self.cross_attention = nn.TransformerDecoder(cross_attn_layer, num_layers=num_cross_layers)

        # 总位移预测头 / Total-displacement prediction head
        self.sequence_pose_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 5),
            nn.Tanh()
        )

    def forward(self,
                depth_sequence: torch.Tensor,
                end_relative_pose: torch.Tensor,
                cfg_mask: Optional[torch.Tensor] = None,
                return_pose_features: bool = False) -> torch.Tensor:

        # 编码深度图像序列与运动 tokens / Encode the depth image sequence and motion tokens
        depth_tokens, motion_tokens = self.depth_encoder(depth_sequence)
        # depth_tokens: (B, N, C)
        # motion_tokens: 启用 use_tea 时为 (B, 3, C)，否则为 None / (B, 3, C) when use_tea is set, else None

        # 编码轨迹终点 (B, 1, C) / Encode the trajectory endpoint, shape (B, 1, C)
        traj_features = self.trajectory_encoder(end_relative_pose).unsqueeze(1)

        # LayerNorm
        enhanced_depth_tokens = self.depth_ln(depth_tokens)
        enhanced_traj_features = self.traj_ln(traj_features)

        # 运动 tokens 的 LayerNorm / LayerNorm for the motion tokens
        if motion_tokens is not None:
            enhanced_motion_tokens = self.motion_ln(motion_tokens)
        else:
            enhanced_motion_tokens = None

        # 添加类型编码（Type Embedding）/ Add the type embedding
        if self.use_type_embed:
            batch_size = enhanced_depth_tokens.shape[0]
            device = enhanced_depth_tokens.device

            # 类型 0: depth tokens / type 0: depth tokens
            num_depth = enhanced_depth_tokens.shape[1]
            type_ids_depth = torch.zeros(batch_size, num_depth, dtype=torch.long, device=device)
            enhanced_depth_tokens = enhanced_depth_tokens + self.type_embed(type_ids_depth)

            # 类型 1: motion tokens / type 1: motion tokens
            if enhanced_motion_tokens is not None:
                type_ids_motion = torch.ones(batch_size, 3, dtype=torch.long, device=device)
                enhanced_motion_tokens = enhanced_motion_tokens + self.type_embed(type_ids_motion)

            # 类型 2: trajectory token / type 2: trajectory token
            type_ids_traj = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            enhanced_traj_features = enhanced_traj_features + self.type_embed(type_ids_traj)

        # CFG 处理 / CFG handling
        if cfg_mask is not None:
            batch_size = enhanced_traj_features.shape[0]
            mask_expanded = cfg_mask.view(batch_size, 1, 1)

            if self.cfg_drop_trajectory:
                enhanced_traj_features = enhanced_traj_features * mask_expanded.float()

            if self.cfg_drop_depth:
                enhanced_depth_tokens = enhanced_depth_tokens * mask_expanded.float()
                # 如果丢弃深度，也丢弃运动信息 / If depth is dropped, also drop the motion info
                if enhanced_motion_tokens is not None:
                    enhanced_motion_tokens = enhanced_motion_tokens * mask_expanded.float()

        # 拼接 depth 与 motion tokens 用于 Self-Attention / Concatenate depth and motion tokens for self-attention
        if enhanced_motion_tokens is not None:
            depth_motion_tokens = torch.cat([enhanced_depth_tokens, enhanced_motion_tokens], dim=1)
        else:
            depth_motion_tokens = enhanced_depth_tokens

        # 两阶段处理 / Two-stage processing:
        # Stage 1: Self-Attention - 视觉 token（含运动）理解障碍物空间关系
        # Stage 1: self-attention - visual tokens (including motion) reason about obstacle spatial relations
        spatial_features = self.self_attention(depth_motion_tokens)

        # Stage 2: Cross-Attention - 注入目标信息 / Stage 2: cross-attention - inject the target information
        transformer_output = self.cross_attention(
            tgt=spatial_features,              # 已理解空间的视觉特征 / visual features that already understand the space
            memory=enhanced_traj_features      # 目标信息 / target information
        )

        # CFG 后处理 / CFG post-processing
        if cfg_mask is not None:
             batch_size = transformer_output.shape[0]
             mask_expanded = cfg_mask.view(batch_size, 1, 1)
             if self.cfg_drop_depth:
                 transformer_output = transformer_output * mask_expanded.float()

        # 关键修改：将目标特征显式拼接到输出序列中，使 UNet 不仅能看到"被目标调制的视觉特征"，
        # 还能直接看到"原始目标指令"，从而有效缓解轨迹左右摇摆，给模型一个稳定的"指南针"
        # Key change: explicitly append the target features to the output sequence so the UNet sees not only the
        # "target-modulated visual features" but also the "raw target instruction", which mitigates trajectory
        # left-right oscillation and gives the model a stable "compass"
        final_sequence = torch.cat([transformer_output, enhanced_traj_features], dim=1)

        if return_pose_features:
            # 位姿预测只依赖视觉特征（不应依赖目标）/ Pose prediction relies only on visual features (it should not depend on the target)
            pose_features = transformer_output.mean(dim=1)
            return final_sequence, pose_features

        return final_sequence

class BSplineDDPM(nn.Module):
    """B-spline 控制点生成的 DDPM 模型，使用官方 UNet1DConditionModel。 / DDPM model for B-spline control point generation, built on the official UNet1DConditionModel."""

    def __init__(self,
                 condition_encoder: ConcatConditionEncoder,
                 num_train_timesteps: int = 1000,
                 fix_first_cp_zero: bool = False,
                 normalizer: Optional[TrajectoryNormalizer] = None,
                 cfg_dropout_prob: float = 0.1,
                 trajectory_interpolation: str = 'bspline',
                 prediction_mode: str = 'control_points',
                 num_control_points: int = 8):
        super().__init__()

        self.condition_encoder = condition_encoder
        self.fix_first_cp_zero = fix_first_cp_zero
        # 若提供，则表示在“归一化空间”训练；推理时自动归一化/反归一化
        # If provided, training happens in the "normalized space"; inference normalizes/denormalizes automatically
        self.normalizer = normalizer
        # CFG 相关参数 / CFG-related parameter
        self.cfg_dropout_prob = cfg_dropout_prob
        # 轨迹插值方法 / Trajectory interpolation method
        self.trajectory_interpolation = trajectory_interpolation
        # 预测模式 / Prediction mode
        self.prediction_mode = prediction_mode
        # B-spline 控制点数量（= UNet 序列长度，训练/推理必须一致）
        # Number of B-spline control points (= UNet sequence length, must match between training and inference)
        self.num_control_points = int(num_control_points)
        # UNet 有 2 次下采样（因子 4），序列长度必须是 4 的倍数，否则 forward 时 skip 连接尺寸不匹配；
        # 提前 fail-fast，给出可读的报错。
        # The UNet has 2 downsampling stages (factor 4), so the sequence length must be a multiple of 4,
        # otherwise the skip-connection sizes mismatch during forward; fail fast with a readable error.
        if self.num_control_points % 4 != 0:
            raise ValueError(
                f"num_control_points={self.num_control_points} 必须是 4 的倍数"
                f"(如 8/12/16/20)。当前 UNet 有 2 次下采样(因子4)，否则会在 forward 时"
                f"报 skip 连接尺寸不匹配。")

        # 根据预测模式初始化 / Initialize according to the prediction mode
        if self.prediction_mode == 'waypoints':
            print(f"🔧 预测模式: Waypoints (固定0.2m间隔)")
        else:
            print(f"🔧 预测模式: Control Points")
            # 初始化轨迹插值器（如果使用 cubic spline）/ Initialize the trajectory interpolator (if cubic spline is used)
            if self.trajectory_interpolation == 'cubic_spline':
                from sand_planner.utils.traj_opt import TrajOpt
                self.traj_opt = TrajOpt()
                print(f"🔧 使用 Cubic Spline 插值生成轨迹")
            else:
                print(f"🔧 使用 B-Spline 插值生成轨迹")

        self.unet = UNet1DConditionModel(
            sample_size=self.num_control_points,  # 目标信号长度（= 控制点数量）/ target signal length (= number of control points)
            in_channels=3,
            out_channels=3,
            layers_per_block=2,  # 每个 UNet block 使用的 ResNet 层数 / number of ResNet layers per UNet block
            # 使用 3 个 stage，确保总下采样次数为 2 次(因子4)与控制点长度对齐
            # Use 3 stages so that the total downsampling (factor 4) stays aligned with the control-point length
            # 优化：减小通道数以防止过拟合和"绕远路"。在 Head Dim 固定为 32 后，模型感知力大增，
            # 不需要过大的通道数即可工作；较小的模型倾向于生成更平滑、更直接的轨迹（正则化效果）。
            # Optimization: reduce channels to prevent overfitting and "detours". With the head dim fixed at 32 the
            # model's perception improves greatly and does not need large channel counts; a smaller model tends to
            # produce smoother, more direct trajectories (a regularization effect).
            block_out_channels=(64, 128, 256),
            # 保持 Head Dim=32，对应 Head 数量为 32/32=1、64/32=2、128/32=4；
            # 这种配置既保证每个 Head 的表达能力，又限制了总体复杂度。
            # Keep head dim=32, giving head counts of 32/32=1, 64/32=2, 128/32=4; this preserves each head's
            # expressiveness while limiting the overall complexity.
            attention_head_dim=(32, 32, 32),
            down_block_types=(
                "DownBlock1D",
                "CrossAttnDownBlock1D",
                "DownBlock1D",
            ),
            mid_block_type="UNetMidBlock1DCrossAttn",
            up_block_types=(
                "ResnetUpsampleBlock1D",
                "CrossAttnUpBlock1D",
                "UpBlock1D",
            ),
            num_class_embeds=None,
            class_embeddings_concat=False,
            # GroupNorm 配置：使用 32 个组以支持 64/128/256 通道数 / GroupNorm config: 32 groups to support 64/128/256 channels
            norm_num_groups=32,
            # 关键修复：显式设置 cross_attention_dim，避免默认的 1280 带来巨大开销
            # Critical fix: set cross_attention_dim explicitly to avoid the default 1280 and its huge overhead
            encoder_hid_dim=condition_encoder.feature_dim,  # 256
            cross_attention_dim=condition_encoder.feature_dim,  # 256
        )

        # DDPM 调度器 / DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,   # 起步通常 1000 或 2000 / typically starts at 1000 or 2000
            beta_schedule="squaredcos_cap_v2",         # 推荐配置 / recommended
            prediction_type="v_prediction",            # 使用 v-prediction 参数化 / use the v-prediction parametrization
            clip_sample=False,
            timestep_spacing="linspace",               # 训练期通常线性 / linear during training
            rescale_betas_zero_snr=True,               # 推荐打开 / recommended on
        )
        self.num_train_timesteps = num_train_timesteps
        
    def forward(self, 
                control_points: torch.Tensor,
                depth_sequence: torch.Tensor,
                end_relative_pose: torch.Tensor,
                total_displacement: Optional[torch.Tensor] = None,
                initial_turn: Optional[torch.Tensor] = None,
                has_initial_turn: Optional[torch.Tensor] = None,
                pose_loss_weight: float = 0.0) -> Dict[str, torch.Tensor]:  # 备选/alt: 0.2 for attention, 0.001 for concat
        """训练前向传播。 / Training forward pass.

        Args:
            control_points: (batch_size, 8, 3) 真实（GT）控制点 / ground-truth control points.
            depth_sequence: (batch_size, seq_len, 1, H, W) 深度图像序列 / depth image sequence.
            end_relative_pose: (batch_size, 3) 轨迹终点相对坐标 / trajectory endpoint relative coordinates.
            total_displacement: (batch_size, 5) 第 1 帧到第 4 帧的总位移 [dx, dy, dz, dyaw, dpitch]（可选）/ total displacement from frame 1 to frame 4 (optional).
            initial_turn: (batch_size,) 初始转向值（可选）/ initial turn value (optional).
            has_initial_turn: (batch_size,) 是否有有效的初始转向（可选）/ whether a valid initial turn exists (optional).
            pose_loss_weight: float，位姿损失权重 / weight of the pose loss.
        Returns:
            loss_dict: 包含损失信息的字典 / a dict containing the loss information.
        """
        batch_size = control_points.size(0)
        device = control_points.device

        # 重塑控制点为 (batch_size, 3, 8) 的 (batch, channels, length) 格式
        # Reshape control points into (batch_size, 3, 8) = (batch, channels, length) layout
        control_points_reshaped = control_points.transpose(1, 2)  # (batch_size, 3, 8)
        # 盒约束：夹回训练归一化盒（[-1, 1]）/ Box constraint: clamp back into the training normalization box ([-1, 1])
        control_points_reshaped = control_points_reshaped.clamp(-1.0, 1.0)

        # 若启用，强制干净样本的第一个控制点为 0（保证数据一致性）
        # If enabled, force the clean sample's first control point to 0 (to keep the data consistent)
        if self.fix_first_cp_zero:
            control_points_reshaped = control_points_reshaped.clone()
            if self.normalizer is not None:
                # 使用归一化空间中的原点值 / Use the origin value in the normalized space
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(control_points_reshaped.device, dtype=control_points_reshaped.dtype)  # (3,)
                control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
            else:
                control_points_reshaped[:, :, 0] = 0.0

        # CFG 训练：随机生成 mask 决定是否丢弃条件 / CFG training: randomly draw a mask to decide whether to drop the condition
        cfg_mask = None
        if self.training and self.cfg_dropout_prob > 0:
            cfg_mask = torch.rand(batch_size, device=device) > self.cfg_dropout_prob

        # 位姿预测分支（如果提供了 GT）/ Pose-prediction branch (if the GT is provided)
        pose_loss = None
        if total_displacement is not None:
            # 使用 return_pose_features 获取中间特征 / Use return_pose_features to obtain the intermediate features
            encoder_hidden_states, pose_features = self.condition_encoder(
                depth_sequence, end_relative_pose,
                initial_turn=initial_turn, has_initial_turn=has_initial_turn,
                cfg_mask=cfg_mask, return_pose_features=True
            )  # encoder_hidden_states: (B, seq_len+1, C), pose_features: (B, C)

            # 通过 MLP 预测位姿 / Predict the pose via the MLP
            predicted_displacement = self.condition_encoder.sequence_pose_predictor(pose_features)  # (B, 5)

            # 计算位姿损失（Huber Loss，对离群值更鲁棒）；beta=1.0：在归一化空间 [-1, 1] 中，|error|<1 时用 MSE，>1 时用 L1
            # Compute the pose loss (Huber loss, more robust to outliers); beta=1.0: in the normalized space [-1, 1],
            # use MSE when |error|<1 and L1 when >1
            pose_loss = F.smooth_l1_loss(predicted_displacement, total_displacement, beta=1.0)
        else:
            # 编码条件：直接返回 token 序列用于 cross-attention / Encode the condition: return the token sequence directly for cross-attention
            encoder_hidden_states = self.condition_encoder(
                depth_sequence, end_relative_pose,
                initial_turn=initial_turn, has_initial_turn=has_initial_turn,
                cfg_mask=cfg_mask
            )  # (batch_size, seq_len+1, feature_dim)

        # 随机采样时间步 / Randomly sample timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()

        # 生成随机噪声 / Sample random noise
        noise = torch.randn_like(control_points_reshaped)

        # 若启用，不对第一个控制点添加噪声（并在损失中屏蔽）
        # If enabled, do not add noise to the first control point (and mask it out in the loss)
        if self.fix_first_cp_zero:
            noise[:, :, 0] = 0.0

        # 添加噪声 / Add the noise
        noisy_control_points = self.scheduler.add_noise(
            control_points_reshaped, noise, timesteps
        )
        if self.fix_first_cp_zero:
            if self.normalizer is not None:
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(noisy_control_points.device, dtype=noisy_control_points.dtype)
                noisy_control_points[:, :, 0] = zero_norm.view(1, 3)
            else:
                noisy_control_points[:, :, 0] = 0.0

        # 预测 v-parametrization：使用官方 UNet 接口，确保与 UNet dtype 对齐
        # Predict the v-parametrization via the official UNet API, ensuring dtype alignment with the UNet
        unet_dtype = self.unet.dtype
        noisy_control_points = noisy_control_points.to(dtype=unet_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
        v_pred = self.unet(
            sample=noisy_control_points,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # 计算 v-parametrization 的目标值；使用 diffusers 自带方法计算 v_target，更准确且与库保持一致
        # Compute the v-parametrization target; use the built-in diffusers method for v_target, which is more
        # accurate and consistent with the library
        v_target = self.scheduler.get_velocity(control_points_reshaped, noise, timesteps)

        # 计算损失 / Compute the loss
        if self.fix_first_cp_zero:
            # 屏蔽第一个控制点对损失的贡献；保证 dtype 一致，并在 float32 中累加以提升数值稳定性
            # Mask out the first control point's contribution to the loss; keep dtype consistent and accumulate in
            # float32 for better numerical stability
            mask = torch.ones_like(v_pred)
            mask[:, :, 0] = 0.0
            diff = (v_pred - v_target.to(dtype=v_pred.dtype)) * mask
            trajectory_loss = (diff.float().pow(2).sum() / mask.float().sum().clamp_min(1.0))
        else:
            trajectory_loss = F.mse_loss(v_pred, v_target.to(dtype=v_pred.dtype))

        # 综合损失：轨迹损失 + 位姿损失 / Combined loss: trajectory loss + pose loss
        if pose_loss is not None:
            total_loss = trajectory_loss + pose_loss_weight * pose_loss
            return {
                'loss': total_loss,
                'trajectory_loss': trajectory_loss,
                'pose_loss': pose_loss,
                'v_pred': v_pred,
                'v_target': v_target,
                'noise': noise
            }
        else:
            return {
                'loss': trajectory_loss,
                'v_pred': v_pred,
                'v_target': v_target,
                'noise': noise
            }
    
    @torch.no_grad()
    def sample(self, 
               depth_sequence: torch.Tensor,
               end_relative_pose: torch.Tensor,
               num_inference_steps: int = 10,
               cfg_scale: float = 1.0,
               initial_turn: Optional[torch.Tensor] = None,
               has_initial_turn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """推理采样（支持 CFG）。 / Inference sampling with CFG support.

        Args:
            depth_sequence: (batch_size, seq_len, 1, H, W) 深度图像序列 / depth image sequence.
            end_relative_pose: (batch_size, 3) 轨迹终点相对坐标 / trajectory endpoint relative coordinates.
            num_inference_steps: 推理步数 / number of inference steps.
            cfg_scale: CFG 引导强度，1.0 表示无引导，>1.0 表示增强条件引导 / CFG guidance scale, 1.0 means no guidance and >1.0 strengthens conditional guidance.
            initial_turn: (batch_size,) 初始转向值（可选，冷启动时为 None 或 has_initial_turn=False）/ initial turn value (optional, None or has_initial_turn=False at cold start).
            has_initial_turn: (batch_size,) 是否有有效的初始转向（可选）/ whether a valid initial turn exists (optional).
        Returns:
            control_points: (batch_size, 8, 3) 生成的控制点 / generated control points.
        """
        # eval 模式 + 统一 device/dtype / Switch to eval mode and unify device/dtype
        self.eval()
        batch_size = depth_sequence.size(0)
        device = next(self.parameters()).device
        unet_dtype = self.unet.dtype

        # 将条件移动到同一 device（编码器内部仍按其精度工作）
        # Move the conditions onto the same device (the encoder still works in its own precision internally)
        depth_sequence = depth_sequence.to(device)

        # 编码条件：返回 token 序列 (batch_size, seq_len+1, feature_dim)；若在归一化空间内训练，则对条件终点做归一化
        # Encode the conditions: returns the token sequence (batch_size, seq_len+1, feature_dim); if trained in the
        # normalized space, normalize the conditioning endpoint
        cond_end = end_relative_pose.to(device)
        if self.normalizer is not None:
            end_np = end_relative_pose.detach().cpu().numpy()
            end_norm_np = self.normalizer.normalize(end_np)
            cond_end = torch.from_numpy(end_norm_np).to(device=device, dtype=end_relative_pose.dtype)

        # 处理 initial_turn 参数 / Handle the initial_turn argument
        cond_initial_turn = None
        cond_has_initial_turn = None
        if initial_turn is not None:
            cond_initial_turn = initial_turn.to(device)
        if has_initial_turn is not None:
            cond_has_initial_turn = has_initial_turn.to(device)
        else:
            # 如果没有提供 has_initial_turn，则根据 initial_turn 是否为 None 判断；推理时 initial_turn=None 表示冷启动
            # If has_initial_turn is not provided, infer it from whether initial_turn is None; at inference,
            # initial_turn=None denotes a cold start
            if cond_initial_turn is None:
                cond_has_initial_turn = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                cond_has_initial_turn = torch.ones(batch_size, dtype=torch.bool, device=device)

        # CFG 推理：准备条件与无条件编码 / CFG inference: prepare the conditional and unconditional encodings
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            # 条件编码（正常条件）/ Conditional encoding (normal condition)
            encoder_hidden_states_cond = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn,
                cfg_mask=torch.ones(batch_size, device=device, dtype=torch.bool)
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

            # 无条件编码（丢弃条件）/ Unconditional encoding (condition dropped)
            encoder_hidden_states_uncond = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn,
                cfg_mask=torch.zeros(batch_size, device=device, dtype=torch.bool)
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

            # 合并为一个批次进行并行推理 / Concatenate into a single batch for parallel inference
            encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
        else:
            # 标准推理（无 CFG）/ Standard inference (no CFG)
            encoder_hidden_states = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

        # 使用 DPM-Solver 多步调度器进行推理（继承训练用 DDPM 调度器配置）
        # Use the DPM-Solver multistep scheduler for inference (inheriting the training DDPM scheduler config)
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config,
            algorithm_type="dpmsolver++",
            prediction_type=self.scheduler.config.prediction_type  # 明确继承 prediction_type / inherit prediction_type explicitly
        )

        dpm_scheduler.set_timesteps(num_inference_steps, device=device)

        # 备选/alt: 在 set_timesteps 阶段启用 Karras sigmas，若当前 diffusers 版本不支持该参数则自动回退
        # Alternative: enable Karras sigmas in set_timesteps, falling back automatically if the current
        # diffusers version does not support the argument
        # karras_enabled = False
        # try:
        #     dpm_scheduler.set_timesteps(num_inference_steps, device=device, use_karras_sigmas=True)

        #     karras_enabled = True
        # except TypeError:
        #     # 旧版本无此参数：尽量设置属性，再使用兼容签名 / older versions lack the arg: set the attribute then use the compatible signature
        #     karras_enabled = False
        #     try:
        #         dpm_scheduler.use_karras_sigmas = True  # 某些版本暴露为属性 / some versions expose it as an attribute
        #         karras_enabled = True
        #     except Exception:
        #         pass
        #     dpm_scheduler.set_timesteps(num_inference_steps, device=device)
        # if not karras_enabled:
        #     print("[WARN] Karras sigmas may not be enabled in this diffusers version.")
        # # 可选调试打印：导出是否启用及首尾若干 sigma/timestep 值 / optional debug print: dump whether enabled and a few head/tail sigma/timestep values
        karras_enabled = False  # 设置默认值 / set the default value
        if os.getenv("DEBUG_SCHEDULER", "0") == "1" or os.getenv("PRINT_SCHEDULER_DEBUG", "0") == "1":
            try:
                if hasattr(dpm_scheduler, "sigmas") and dpm_scheduler.sigmas is not None:
                    import numpy as _np
                    _s = _np.asarray(dpm_scheduler.sigmas, dtype=_np.float32)
                    _head = _s[:3].tolist() if _s.size >= 3 else _s.tolist()
                    _tail = _s[-3:].tolist() if _s.size >= 3 else _s.tolist()
                    print(f"[Scheduler] Karras enabled={karras_enabled}, sigmas shape={_s.shape}, head={_head}, tail={_tail}")
                else:
                    _ts = dpm_scheduler.timesteps
                    try:
                        _ts_list = _ts[:3].tolist(), _ts[-3:].tolist()
                    except Exception:
                        _ts_list = (list(_ts)[:3], list(_ts)[-3:])
                    print(f"[Scheduler] Karras enabled={karras_enabled}, timesteps head={_ts_list[0]}, tail={_ts_list[1]}")
            except Exception:
                # 调试信息失败不影响推理 / A failure in the debug output must not affect inference
                pass

        # 初始噪声，(batch_size, 3, num_control_points) 格式 / Initial noise in (batch_size, 3, num_control_points) layout
        control_points_reshaped = torch.randn(batch_size, 3, self.num_control_points, device=device, dtype=unet_dtype)
        if self.fix_first_cp_zero:
            if self.normalizer is not None:
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(device=device, dtype=control_points_reshaped.dtype)
                control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
            else:
                control_points_reshaped[:, :, 0] = 0.0

        # 去噪过程 / Denoising loop
        for t in tqdm(dpm_scheduler.timesteps, desc="Sampling"):
            if use_cfg:
                # CFG 推理：并行预测条件与无条件 / CFG inference: predict conditional and unconditional in parallel
                timesteps = torch.full((batch_size * 2,), t, device=device, dtype=torch.long)

                # 复制输入用于并行推理 / Duplicate the input for parallel inference
                model_in_duplicated = torch.cat([control_points_reshaped, control_points_reshaped], dim=0)
                model_in_duplicated = dpm_scheduler.scale_model_input(model_in_duplicated, t)

                # 预测 v-parametrization（并行）/ Predict the v-parametrization (in parallel)
                v_pred_combined = self.unet(
                    sample=model_in_duplicated,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # 分离条件与无条件预测 / Split the conditional and unconditional predictions
                v_pred_uncond, v_pred_cond = v_pred_combined.chunk(2, dim=0)

                # CFG: v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
            else:
                # 标准推理（无 CFG）/ Standard inference (no CFG)
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_in = dpm_scheduler.scale_model_input(control_points_reshaped, t)
                v_pred = self.unet(
                    sample=model_in,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            # 去噪步骤 / Denoising step
            control_points_reshaped = dpm_scheduler.step(
                v_pred, t, control_points_reshaped
            ).prev_sample
            if self.fix_first_cp_zero:
                if self.normalizer is not None:
                    zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(device=device, dtype=control_points_reshaped.dtype)
                    control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
                else:
                    control_points_reshaped[:, :, 0] = 0.0

        # 重塑为控制点形状 (batch_size, 8, 3) / Reshape into the control-point shape (batch_size, 8, 3)
        control_points = control_points_reshaped.transpose(1, 2)

        # 若训练在归一化空间内，则将输出反归一化到原尺度
        # If trained in the normalized space, denormalize the output back to the original scale
        if self.normalizer is not None:
            # 归一化空间一般在 [-1, 1]（percentile/zscore 裁剪）；推理阶段先夹紧再反归一化可避免外推放大
            # The normalized space usually lies in [-1, 1] (percentile/zscore clipping); clamping before
            # denormalizing at inference avoids amplifying extrapolation
            control_points = control_points.clamp(-1.0, 1.0)
            cp_np = control_points.detach().cpu().numpy()
            cp_denorm_np = self.normalizer.denormalize(cp_np)
            control_points = torch.from_numpy(cp_denorm_np).to(control_points.device, dtype=control_points.dtype)

        return control_points

def create_model(args) -> BSplineDDPM:
    """创建模型。 / Create the model."""

    # 1. 条件编码器 / Condition encoder
    print("创建条件编码器...")
    fusion_strategy = getattr(args, 'multi_frame_fusion', 'concat')
    use_tea = getattr(args, 'use_tea', False)
    use_type_embed = getattr(args, 'use_type_embed', True)
    use_initial_turn = getattr(args, 'use_initial_turn', False)
    
    print(f"🔄 多帧融合策略: {fusion_strategy}")
    if use_tea:
        print(f"🔥 TEA-lite 已启用: 将从 {args.sequence_length} 帧生成 {args.sequence_length-1} 个运动差分token")
    if use_type_embed:
        print(f"🎯 Type Embedding 已启用")
    if use_initial_turn:
        print(f"🔄 Initial Turn 已启用: 使用 (CP[1]+CP[2])/2 的 Y 分量作为转向条件")
    
    # 使用 Concat/Cross Attention 融合目标与视觉特征 / Fuse the target with the visual features via concat/cross attention
    print("🚀 使用 Concat Attention 融合目标与视觉特征")
    condition_encoder = ConcatConditionEncoder(
        feature_dim=args.feature_dim,
        seq_len=args.sequence_length,
        num_transformer_layers=args.transformer_layers,
        num_heads=args.transformer_heads,
        cfg_drop_depth=getattr(args, 'cfg_drop_depth', False),
        cfg_drop_trajectory=getattr(args, 'cfg_drop_trajectory', True),
        fusion_strategy=fusion_strategy,
        use_tea=use_tea,  # 传递 TEA 参数 / pass through the TEA flag
        use_type_embed=use_type_embed,  # 传递 Type Embedding 参数 / pass through the type-embedding flag
        use_initial_turn=use_initial_turn  # 传递 Initial Turn 参数 / pass through the initial-turn flag
    )

    # 2. 创建 DDPM 模型 / Create the DDPM model
    normalizer = None
    if getattr(args, 'normalize_targets', False):
        try:
            normalizer = TrajectoryNormalizer(
                args.stats_json_path,
                method=args.norm_method,
                margin=args.norm_margin,
                clamp=getattr(args, 'norm_clamp', False),
            )
        except Exception as e:
            print(f"[Warn] Failed to load normalizer: {e}")
            normalizer = None

    model = BSplineDDPM(
        condition_encoder=condition_encoder,
        num_train_timesteps=args.num_train_timesteps,
        fix_first_cp_zero=getattr(args, 'fix_first_cp_zero', False),
        normalizer=normalizer,
        cfg_dropout_prob=getattr(args, 'cfg_dropout_prob', 0.1),
        trajectory_interpolation=getattr(args, 'trajectory_interpolation', 'bspline'),
        prediction_mode=getattr(args, 'prediction_mode', 'control_points'),
        num_control_points=getattr(args, 'num_control_points', 8),
    )

    return model

def train_epoch(model: BSplineDDPM, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                accelerator: Accelerator,
                start_step: int = 0,
                wandb_log_interval: int = 50) -> Tuple[Dict[str, float], int]:
    """训练一个 epoch。 / Train for one epoch."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    step = start_step

    progress_bar = tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        # 获取数据 / Fetch the data
        depth_sequence = batch['depth_sequence']
        # 默认优先使用归一化字段（若存在）/ Prefer the normalized fields when present
        control_points = batch['control_points_norm'] if 'control_points_norm' in batch else batch['control_points']
        end_relative_pose = batch['end_relative_pose_norm'] if 'end_relative_pose_norm' in batch else batch['end_relative_pose']
        # 优先使用归一化的 total_displacement / Prefer the normalized total_displacement
        total_displacement = batch.get('total_displacement_norm', batch.get('total_displacement', None))
        # 获取初始转向信息 / Fetch the initial-turn information
        initial_turn = batch.get('initial_turn', None)
        has_initial_turn = batch.get('has_initial_turn', None)

        # 前向传播 / Forward pass
        with accelerator.accumulate(model):
            loss_dict = model(control_points, depth_sequence, end_relative_pose,
                             total_displacement=total_displacement,
                             initial_turn=initial_turn, has_initial_turn=has_initial_turn)
            loss = loss_dict['loss']

            # 反向传播 / Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        step += 1

        # 更新进度条，附带显示位姿损失 / Update the progress bar, also showing the pose loss
        postfix_dict = {
            'loss': f"{loss.item():.6f}",
            'avg_loss': f"{total_loss / num_batches:.6f}"
        }
        if 'pose_loss' in loss_dict:
            postfix_dict['pose_loss'] = f"{loss_dict['pose_loss'].item():.6f}"
        progress_bar.set_postfix(postfix_dict)

        # 记录到跟踪器（降低频率以减少 I/O 开销）/ Log to the tracker (reduced frequency to lower I/O overhead)
        if wandb_log_interval == 0 or step % wandb_log_interval == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/avg_loss': total_loss / num_batches,
            }
            if 'trajectory_loss' in loss_dict:
                log_dict['train/trajectory_loss'] = loss_dict['trajectory_loss'].item()
            if 'pose_loss' in loss_dict:
                log_dict['train/pose_loss'] = loss_dict['pose_loss'].item()
            accelerator.log(log_dict, step=step)
    
    return {'avg_loss': total_loss / num_batches}, step

@torch.no_grad()
def validate_epoch(model: BSplineDDPM, 
                   dataloader: DataLoader, 
                   accelerator: Accelerator,
                   current_step: Optional[int] = None) -> Dict[str, float]:
    """验证一个 epoch。 / Validate for one epoch."""

    model.eval()
    total_loss = 0.0
    num_batches = 0
    # 采样对比：收集前若干样本的 预测 CP vs GT CP / Sampling comparison: collect predicted CP vs GT CP for the first few samples
    cp_examples = []  # list of dict{run_id, start, end, pred_cp(list), gt_cp(list), mae, maxerr}
    max_examples = 50
    sample_steps = 10  # 采样步数：适中速度 / number of sampling steps: a moderate-speed setting
    # 控制点 RMSE 累计（原尺度）：全局 RMSE = sqrt(sum((pred-gt)^2) / N_elements)
    # Control-point RMSE accumulators (original scale): global RMSE = sqrt(sum((pred-gt)^2) / N_elements)
    cp_se_sum = 0.0
    cp_elem_count = 0

    progress_bar = tqdm(dataloader, desc="Validation", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        depth_sequence = batch['depth_sequence']
        control_points = batch['control_points_norm'] if 'control_points_norm' in batch else batch['control_points']
        end_relative_pose = batch['end_relative_pose_norm'] if 'end_relative_pose_norm' in batch else batch['end_relative_pose']
        # 优先使用归一化的 total_displacement / Prefer the normalized total_displacement
        total_displacement = batch.get('total_displacement_norm', batch.get('total_displacement', None))
        # 获取初始转向信息 / Fetch the initial-turn information
        initial_turn = batch.get('initial_turn', None)
        has_initial_turn = batch.get('has_initial_turn', None)

    # 前向传播（计算损失）/ Forward pass (compute the loss)
        loss_dict = model(control_points, depth_sequence, end_relative_pose,
                         total_displacement=total_displacement,
                         initial_turn=initial_turn, has_initial_turn=has_initial_turn)
        loss = loss_dict['loss']

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条以显示位姿损失 / Update the progress bar to show the pose loss
        postfix_dict = {
            'val_loss': f"{loss.item():.6f}",
            'avg_val_loss': f"{total_loss / num_batches:.6f}"
        }
        if 'pose_loss' in loss_dict:
            postfix_dict['pose_loss'] = f"{loss_dict['pose_loss'].item():.6f}"
        progress_bar.set_postfix(postfix_dict)

        # 采样对比 + RMSE：采样一次以累计 RMSE，并在样本上限内收集表格示例
        # Sampling comparison + RMSE: sample once to accumulate RMSE and collect table examples up to the cap
        try:
            # 使用原始尺度的条件进行采样，模型内部会自动归一化/反归一化
            # Sample with the original-scale conditions; the model normalizes/denormalizes internally
            end_pose_for_cond = batch['end_relative_pose']  # 直接使用原尺度条件 / use the original-scale condition directly

            cp_pred = model.sample(depth_sequence, end_pose_for_cond, num_inference_steps=sample_steps)  # (B,8,3) 原尺度 / original scale
            # 确保 GT 也是原尺度：优先使用原始控制点，而非归一化版本
            # Ensure the GT is also at the original scale: prefer the raw control points over the normalized ones
            cp_gt = batch['control_points']  # (B,8,3) 原尺度 / original scale

            # RMSE 累计（按元素）：cp_pred 和 cp_gt 都是原尺度 / Accumulate RMSE element-wise: both cp_pred and cp_gt are at the original scale
            diff_cpu = (cp_pred.detach().cpu() - cp_gt.detach().cpu())  # (B,8,3)
            cp_se_sum += float((diff_cpu.pow(2)).sum().item())
            cp_elem_count += int(diff_cpu.numel())

            # 表格示例（仅收集到上限条数）/ Table examples (collected only up to the cap)
            remaining = max_examples - len(cp_examples)
            if remaining > 0:
                bsz = cp_gt.shape[0]
                take = min(bsz, remaining)
                run_ids = batch.get('run_ids', [None]*bsz)
                starts = batch.get('start_indices', torch.zeros(bsz, dtype=torch.long)).tolist() if 'start_indices' in batch else [None]*bsz
                ends = batch.get('end_indices', torch.zeros(bsz, dtype=torch.long)).tolist() if 'end_indices' in batch else [None]*bsz

                cp_pred_np = cp_pred.detach().cpu().numpy()
                cp_gt_np = cp_gt.detach().cpu().numpy()
                for i in range(take):
                    pred = cp_pred_np[i]
                    gt = cp_gt_np[i]
                    diff = pred - gt
                    dists = (diff**2).sum(axis=-1) ** 0.5  # (8,)
                    mae = float(abs(diff).mean())
                    maxerr = float(dists.max())
                    cp_examples.append({
                        'run_id': run_ids[i] if isinstance(run_ids, list) else None,
                        'start': starts[i] if isinstance(starts, list) else None,
                        'end': ends[i] if isinstance(ends, list) else None,
                        'pred_cp': pred.tolist(),
                        'gt_cp': gt.tolist(),
                        'mae': mae,
                        'maxerr': maxerr,
                    })
        except Exception:
            # 不影响主验证流程 / Must not affect the main validation flow
            pass

    avg_loss = total_loss / num_batches

    # 记录损失与 RMSE 到跟踪器 / Log the loss and RMSE to the tracker
    log_dict = {'val/loss': avg_loss}
    if cp_elem_count > 0:
        cp_rmse = float(math.sqrt(cp_se_sum / cp_elem_count))
        log_dict['val/cp_rmse'] = cp_rmse
    # 验证日志通常每个 epoch 一次，频率已经合理 / Validation logs run roughly once per epoch, which is a reasonable frequency
    accelerator.log(log_dict, step=current_step)

    # 记录对比表到 W&B（若启用）/ Log the comparison table to W&B (if enabled)
    try:
        if cp_examples and accelerator.trackers is not None:
            # 仅在 W&B 模式下创建表格 / Only build the table in W&B mode
            if any(getattr(t, 'name', '').startswith('wandb') for t in accelerator.trackers):
                import wandb
                table = wandb.Table(columns=['idx', 'run_id', 'start', 'end', 'mae', 'maxerr', 'pred_cp', 'gt_cp'])
                for idx, rec in enumerate(cp_examples):
                    table.add_data(idx, rec['run_id'], rec['start'], rec['end'], rec['mae'], rec['maxerr'], rec['pred_cp'], rec['gt_cp'])
                accelerator.log({'val/cp_compare': table}, step=current_step)
    except Exception:
        pass
    
    metrics = {'avg_loss': avg_loss}
    if cp_elem_count > 0:
        metrics['cp_rmse'] = float(math.sqrt(cp_se_sum / cp_elem_count))
    return metrics


def save_checkpoint(model: BSplineDDPM, 
                   optimizer: optim.Optimizer, 
                   epoch: int, 
                   best_loss: float,
                   save_path: str,
                   lr_scheduler=None):
    """保存检查点。 / Save a checkpoint."""

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        # 记录控制点数量（UNet 序列长度）：1D UNet 权重与长度无关，加载时无法从权重推断，
        # 必须显式保存以供推理端自动读取，避免静默不一致。
        # Record the number of control points (UNet sequence length): 1D UNet weights are independent of the
        # length and cannot be inferred from them on load, so it must be saved explicitly for the inference side
        # to read automatically, avoiding silent mismatches.
        'num_control_points': int(getattr(model, 'num_control_points', 8)),
    }

    # 保存学习率调度器状态 / Save the learning-rate scheduler state
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='B-Spline DDPM Training')

    # 数据参数 / Data arguments
    parser.add_argument('--dataset_root', type=str, default='dataset',
                       help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=4, 
                       help='Depth sequence length')
    parser.add_argument('--image_height', type=int, default=168,
                       help='Depth image height after preprocessing')
    parser.add_argument('--image_width', type=int, default=224,
                       help='Depth image width after preprocessing')
    parser.add_argument('--downscale_depth', action='store_true',
                       help='First downscale depth 640x480 to 320x240 before any resize')
    parser.add_argument('--min_gap', type=int, default=3, 
                       help='Minimum gap for trajectory')
    parser.add_argument('--max_gap', type=int, default=25, 
                       help='Maximum gap for trajectory')
    parser.add_argument('--min_trajectory_length', type=int, default=3,
                       help='Minimum trajectory length required for B-spline fitting (clamps min_gap)')
    parser.add_argument('--max_arc_length', type=float, default=None,
                       help='Maximum trajectory arc length in meters (e.g., 3.0). Longer trajectories are truncated before fitting.')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--samples_per_epoch', type=int, default=6000,
                       help='Number of training samples per epoch (controls steps per epoch)')
    # 归一化配置 / Normalization configuration
    parser.add_argument('--normalize_targets', action='store_true',
                       help='Use dataset-wide stats to normalize targets (control points and end_relative_pose)')
    parser.add_argument('--stats_json_path', type=str, default='outputs/my_stats.json',
                       help='Path to stats JSON produced by compute_trajectory_stats.py')
    parser.add_argument('--norm_method', type=str, default='percentile', choices=['percentile','zscore'],
                       help='Normalization method for targets')
    parser.add_argument('--norm_margin', type=float, default=0.10,
                       help='Extra margin ratio added to percentile range (e.g., 0.10 => 10 percent)')
    parser.add_argument('--norm_clamp', action='store_true',
                       help='Clamp normalized values to range ([-1,1] for percentile or +/-z_clip for zscore)')
    
    # 静态序列数据增强参数 / Static-sequence data augmentation arguments
    parser.add_argument('--static_sequence_prob', type=float, default=0.0,
                       help='Probability of applying static sequence augmentation (0.0 to 1.0)')
    parser.add_argument('--static_sequence_strategy', type=str, default='mixed',
                       choices=['full', 'partial', 'mixed'],
                       help='Static sequence augmentation strategy')
    parser.add_argument('--static_frame_selection', type=str, default='last',
                       choices=['first', 'middle', 'last', 'random'],
                       help='Which frame to use for static sequence')
    
    # 多帧融合策略 / Multi-frame fusion strategy
    parser.add_argument('--multi_frame_fusion', type=str, default='concat',
                       choices=['concat', 'average', 'attention'],
                       help='Multi-frame fusion strategy: concat (原始拼接), average (时序平均), attention (注意力融合)')
    parser.add_argument('--max_tokens_per_frame', type=int, default=96,
                       help='Maximum spatial tokens per frame (8x12=96 for stride=16 features)')
    
    # 模型参数 / Model arguments
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--transformer_layers', type=int, default=2, 
                       help='Number of transformer layers')
    parser.add_argument('--transformer_heads', type=int, default=4, 
                       help='Number of transformer attention heads')
    parser.add_argument('--num_train_timesteps', type=int, default=1000, 
                       help='Number of DDPM training timesteps')
    parser.add_argument('--fix_first_cp_zero', action='store_true',
                       help='Fix the first control point to (0,0,0), no noise and masked loss on that position')

    # TEA-lite 参数 / TEA-lite arguments
    parser.add_argument('--use_tea', action='store_true',
                       help='启用TEA-lite运动编码（从4帧深度序列提取3个帧间差分token）')

    # 轨迹插值方法 / Trajectory interpolation method
    parser.add_argument('--trajectory_interpolation', type=str, default='bspline',
                       choices=['bspline', 'cubic_spline'],
                       help='轨迹插值方法: bspline (B样条), cubic_spline (三次样条)')

    # 预测模式 / Prediction mode
    parser.add_argument('--prediction_mode', type=str, default='control_points',
                       choices=['control_points', 'waypoints'],
                       help='预测模式: control_points (控制点+样条拟合), waypoints (固定0.2m间隔)')

    # B-spline 控制点数量（= UNet 序列长度，训练/推理必须一致）/ Number of B-spline control points (= UNet sequence length, must match between training and inference)
    parser.add_argument('--num_control_points', type=int, default=8,
                       help='B样条控制点数量(同时是UNet输出序列长度)；改此值需重新训练，推理端必须用同样的值')

    # Type Embedding 参数 / Type-embedding arguments
    parser.add_argument('--use_type_embed', action='store_true', default=True,
                       help='启用类型编码，区分depth/motion/trajectory三种模态的token')
    parser.add_argument('--no_type_embed', dest='use_type_embed', action='store_false',
                       help='禁用类型编码')

    # Initial Turn 参数 / Initial-turn arguments
    parser.add_argument('--use_initial_turn', action='store_true',
                       help='启用初始转向条件（从GT控制点提取转向方向）')

    # CFG 参数 / CFG arguments
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.0,
                       help='CFG条件丢弃概率 (0.0-1.0)，用于Classifier-Free Guidance训练')
    parser.add_argument('--cfg_drop_depth', action='store_true',
                       help='CFG时是否丢弃深度条件')
    parser.add_argument('--cfg_drop_trajectory', type=bool, default=True,
                       help='CFG时是否丢弃轨迹条件（默认True，仅丢弃轨迹）')
    parser.add_argument('--no_cfg_drop_trajectory', dest='cfg_drop_trajectory', action='store_false',
                       help='不丢弃轨迹条件（与--cfg_drop_depth配合实现仅丢弃深度）')

    # 训练参数 / Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                       help='Weight decay')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                       choices=['no', 'fp16', 'bf16'],
                       help='Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).')

    # 学习率调度 / Learning-rate schedule
    parser.add_argument('--use_lr_scheduler', action='store_true',
                       help='使用余弦学习率调度')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'linear', 'exponential'],
                       help='学习率调度器类型')
    parser.add_argument('--lr_warmup_epochs', type=int, default=10,
                       help='学习率预热轮数')
    parser.add_argument('--lr_min_factor', type=float, default=0.01,
                       help='最小学习率因子（相对于初始学习率）')

    # 保存和日志 / Saving and logging
    parser.add_argument('--output_dir', type=str, default='./checkpoints/concat1',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='断点续训：checkpoint .pth 路径，恢复模型/优化器/调度器/epoch')
    parser.add_argument('--log_interval', type=int, default=100, 
                       help='Log interval')
    parser.add_argument('--save_interval', type=int, default=10, 
                       help='Save interval')
    parser.add_argument('--wandb_log_interval', type=int, default=50,
                       help='W&B logging interval (steps), 0 means log every step')
    parser.add_argument('--wandb_project', type=str, default='bspline-ddpm', 
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Optional W&B run name')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--one_batch_only', action='store_true',
                       help='If set, dataloader yields exactly one batch (debug mode)')
    # 采样验证已并入 validate_epoch，仅用于 CP 对比与 W&B 记录，不再单独配置
    # Sampling-based validation is folded into validate_epoch and is only used for CP comparison and W&B
    # logging, so it is no longer configured separately

    args = parser.parse_args()

    # 创建输出目录 / Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 accelerator 与 W&B（可关闭）/ Initialize accelerator and W&B (can be disabled)
    log_with = "wandb" if not getattr(args, 'disable_wandb', False) else None
    accelerator = Accelerator(log_with=log_with, mixed_precision=args.mixed_precision)
    if log_with == "wandb":
        init_kwargs = {"wandb": {"settings":wandb.Settings(start_method="thread", _service_wait=300)}}
        if getattr(args, 'wandb_run_name', None):
            init_kwargs["wandb"]["name"] = args.wandb_run_name
        accelerator.init_trackers(args.wandb_project, config=vars(args), init_kwargs=init_kwargs)
    
    # 根据实际数据集调整权重，更新为实际存在的数据集 / Adjust the weights for the actual datasets that exist
    scene_sampling_weights = {
        'dataset_avoid': 0.3,                 # 避障场景 / obstacle-avoidance scene (151 runs, 42.2%)
        'dataset_chair_cross_human': 0.20,     # 椅子交叉人员场景 / chair-crossing-human scene (90 runs, 25.1%)
        'dataset_ccts': 0.20,                  # CCTS 场景 / CCTS scene (117 runs, 32.7%)
        'dataset_pure_stair': 0.1
    }
    print(f"🏢 使用实际数据集的场景权重配置:")
    print(f"📊 详细场景权重配置:")
    for scene, weight in scene_sampling_weights.items():
        print(f"   {scene}: {weight:.1%}")

    # 静态序列增强信息 / Static-sequence augmentation info
    if args.static_sequence_prob > 0:
        print(f"🔄 启用静态序列数据增强:")
        print(f"   概率: {args.static_sequence_prob:.1%}")
        print(f"   策略: {args.static_sequence_strategy}")
        print(f"   帧选择: {args.static_frame_selection}")
    else:
        print(f"❌ 静态序列数据增强已禁用")

    # CFG 信息 / CFG info
    if args.cfg_dropout_prob > 0:
        print(f"🎯 启用CFG (Classifier-Free Guidance):")
        print(f"   条件丢弃概率: {args.cfg_dropout_prob:.1%}")
        print(f"   轨迹条件丢弃: ✅ (总是)")
        print(f"   深度条件丢弃: {'✅' if args.cfg_drop_depth else '❌'}")
        print(f"   LayerNorm修正: ✅ (先LN再mask，避免偏置泄漏)")
        if not args.cfg_drop_depth:
            print(f"   📝 当前仅丢弃轨迹条件，保留深度感知能力")
        else:
            print(f"   📝 同时丢弃深度和轨迹条件，更强的无条件生成")
    else:
        print(f"❌ CFG已禁用 (cfg_dropout_prob=0)")

    # 创建数据加载器 / Create the data loaders
    train_dataloader = create_bspline_dataloader(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        num_workers=args.num_workers,
    shuffle=True,
    one_batch_only=args.one_batch_only,
    samples_per_epoch=args.samples_per_epoch,
    min_trajectory_length=args.min_trajectory_length,
    # 图像预处理 / Image preprocessing
    image_size=(args.image_height, args.image_width),
    downscale_depth_half=args.downscale_depth,
    # 目标归一化透传 / Pass-through of the target normalization settings
    normalize_targets=args.normalize_targets,
    stats_json_path=args.stats_json_path,
    norm_method=args.norm_method,
    norm_margin=args.norm_margin,
    norm_clamp=args.norm_clamp,
    # 静态序列数据增强参数（仅训练集使用）/ Static-sequence data augmentation (training set only)
    static_sequence_prob=args.static_sequence_prob,
    static_sequence_strategy=args.static_sequence_strategy,
    static_frame_selection=args.static_frame_selection,
    # 楼梯场景权重优化 / Stair-scene sampling-weight tuning
    scene_sampling_weights=scene_sampling_weights,
    # 训练/验证集分割 / Train/validation split
    split_mode="train",
    train_ratio=0.95,
    random_seed=19980816,
    # 轨迹插值方法 / Trajectory interpolation method
    trajectory_interpolation=args.trajectory_interpolation,
    # 预测模式 / Prediction mode
    prediction_mode=args.prediction_mode,
    # 控制点数量 / Number of control points
    num_control_points=args.num_control_points,
    max_arc_length=args.max_arc_length,
    )

    val_dataloader = create_bspline_dataloader(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        min_gap=4,
        max_gap=args.max_gap,
        num_workers=args.num_workers,
    shuffle=False,
    one_batch_only=args.one_batch_only,
    samples_per_epoch=max(args.batch_size, args.samples_per_epoch // 10),
    min_trajectory_length=args.min_trajectory_length,
    image_size=(args.image_height, args.image_width),
    downscale_depth_half=args.downscale_depth,
    normalize_targets=args.normalize_targets,
    stats_json_path=args.stats_json_path,
    norm_method=args.norm_method,
    norm_margin=args.norm_margin,
    norm_clamp=args.norm_clamp,
    # 验证集使用相同的静态序列增强概率来验证性能 / The validation set uses the same static-sequence augmentation probability to gauge performance
    static_sequence_prob=args.static_sequence_prob,
    static_sequence_strategy=args.static_sequence_strategy,
    static_frame_selection=args.static_frame_selection,
    # 验证集也使用相同的场景权重 / The validation set also uses the same scene weights
    scene_sampling_weights=scene_sampling_weights,
    # 训练/验证集分割 / Train/validation split
    split_mode="val",
    train_ratio=0.95,
    random_seed=19980816,
    # 轨迹插值方法 / Trajectory interpolation method
    trajectory_interpolation=args.trajectory_interpolation,
    # 预测模式 / Prediction mode
    prediction_mode=args.prediction_mode,
    # 控制点数量 / Number of control points
    num_control_points=args.num_control_points,
    max_arc_length=args.max_arc_length,
    )

    # 创建模型 / Create the model
    model = create_model(args)

    # 设置差异化学习率：可给骨干网络更小的学习率，其他部分使用正常学习率
    # Set up differentiated learning rates: the backbone can use a smaller LR while the rest uses the normal LR
    backbone_lr = args.learning_rate  # 此处骨干网络学习率与正常学习率相同 / here the backbone LR equals the normal LR
    normal_lr = args.learning_rate

    # 分离参数组 / Split the parameter groups
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'condition_encoder.depth_encoder.backbone' in name:
            backbone_params.append(param)
            print(f"骨干网络参数: {name} (lr={backbone_lr:.2e})")
        else:
            other_params.append(param)

    print(f"骨干网络参数数量: {len(backbone_params)}")
    print(f"其他参数数量: {len(other_params)}")
    print(f"骨干网络学习率: {backbone_lr:.2e}")
    print(f"其他部分学习率: {normal_lr:.2e}")

    # 创建参数组 / Build the parameter groups
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': normal_lr,
            'name': 'other'
        })
    
    # 创建优化器 / Create the optimizer
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器 / Create the learning-rate scheduler
    lr_scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler_type == 'cosine':
            # 余弦退火调度器 / Cosine annealing scheduler
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.learning_rate * args.lr_min_factor
            )
        elif args.lr_scheduler_type == 'linear':
            # 线性衰减调度器 / Linear decay scheduler
            lr_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=args.lr_min_factor,
                total_iters=args.epochs
            )
        elif args.lr_scheduler_type == 'exponential':
            # 指数衰减调度器 / Exponential decay scheduler
            gamma = (args.lr_min_factor) ** (1.0 / args.epochs)
            lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        accelerator.print(f"📊 学习率调度器: {args.lr_scheduler_type}")
        accelerator.print(f"   初始学习率: {args.learning_rate}")
        accelerator.print(f"   最小学习率: {args.learning_rate * args.lr_min_factor}")
        accelerator.print(f"   预热轮数: {args.lr_warmup_epochs}")
    else:
        accelerator.print(f"📊 学习率调度: 固定学习率 {args.learning_rate}")

    # 使用 accelerator 准备模型与数据 / Prepare the model and data with accelerator
    if lr_scheduler is not None:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    
    # 训练循环 / Training loop
    best_val_loss = float('inf')
    start_epoch = 0

    # 断点续训：恢复模型/优化器/调度器/epoch / Resume training: restore model/optimizer/scheduler/epoch
    if getattr(args, 'resume', None):
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume 指定的 checkpoint 不存在: {args.resume}")
        accelerator.print(f"🔁 从 checkpoint 续训: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if lr_scheduler is not None and ckpt.get('lr_scheduler_state_dict') is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_val_loss = float(ckpt.get('best_loss', best_val_loss))
        accelerator.print(f"   恢复完成：start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

    accelerator.print(f"Starting training for {args.epochs} epochs (from epoch {start_epoch})...")
    try:
        steps_per_epoch = len(train_dataloader)
        accelerator.print(f"len(train_dataloader) = {steps_per_epoch} (one_batch_only={args.one_batch_only})")
        accelerator.print(f"Steps/epoch = {steps_per_epoch}, approx samples/epoch = {steps_per_epoch * args.batch_size}")
    except Exception:
        pass
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        accelerator.print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练 / Train
        train_metrics, global_step = train_epoch(
            model, train_dataloader, optimizer, accelerator,
            start_step=global_step, wandb_log_interval=args.wandb_log_interval
        )

        # 验证 / Validate
        val_metrics = validate_epoch(model, val_dataloader, accelerator, current_step=global_step)
    # 采样验证已在 validate_epoch 内完成 CP 对比与 W&B 记录，不再单独执行
    # Sampling-based validation already does the CP comparison and W&B logging inside validate_epoch, so it is not run separately

        # 学习率调度，在验证后更新 / Learning-rate schedule, updated after validation
        if lr_scheduler is not None:
            # 应用预热策略 / Apply the warmup strategy
            if epoch < args.lr_warmup_epochs:
                # 预热阶段：线性增长到目标学习率 / Warmup phase: linearly ramp up to the target LR
                warmup_factor = (epoch + 1) / args.lr_warmup_epochs
                for i, param_group in enumerate(optimizer.param_groups):
                    if 'name' in param_group and param_group['name'] == 'backbone':
                        target_lr = backbone_lr * warmup_factor
                    else:
                        target_lr = args.learning_rate * warmup_factor
                    param_group['lr'] = target_lr

                # 显示所有参数组的学习率 / Show the LR of every parameter group
                lr_info = []
                for param_group in optimizer.param_groups:
                    name = param_group.get('name', 'default')
                    lr = param_group['lr']
                    lr_info.append(f"{name}={lr:.2e}")
                accelerator.print(f"🔥 Warmup: epoch {epoch+1}/{args.lr_warmup_epochs}, lr = {', '.join(lr_info)}")
            else:
                # 正常调度阶段，先记录调度前的学习率 / Normal schedule phase: record the LRs before stepping
                old_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                lr_scheduler.step()

                # 检查是否有学习率变化 / Check whether the LR changed
                new_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                lr_changed = any(abs(new_lr - old_lr) > 1e-8 for new_lr, old_lr in zip(new_lrs, old_lrs))

                if lr_changed:
                    lr_info = []
                    for param_group in optimizer.param_groups:
                        name = param_group.get('name', 'default')
                        lr = param_group['lr']
                        lr_info.append(f"{name}={lr:.2e}")
                    accelerator.print(f"📊 LR Scheduler: {', '.join(lr_info)}")

        # 显示当前学习率 / Show the current learning rates
        current_lrs = []
        for param_group in optimizer.param_groups:
            name = param_group.get('name', 'default')
            lr = param_group['lr']
            current_lrs.append(f"{name}={lr:.2e}")
        current_lr_display = ', '.join(current_lrs)

        # 记录 / Logging
        if accelerator.is_local_main_process:
            accelerator.print(f"Train Loss: {train_metrics['avg_loss']:.6f}")
            accelerator.print(f"Val Loss: {val_metrics['avg_loss']:.6f}")
            accelerator.print(f"Learning Rates: {current_lr_display}")
            if 'cp_rmse' in val_metrics:
                accelerator.print(f"Val CP RMSE: {val_metrics['cp_rmse']:.6f}")
        
        # 构建 wandb 日志字典 / Build the wandb log dict
        wandb_log = {
            'epoch': epoch + 1,
            'train/epoch_loss': train_metrics['avg_loss'],
            'val/epoch_loss': val_metrics['avg_loss'],
        }

        # 记录所有参数组的学习率 / Log the LR of every parameter group
        for param_group in optimizer.param_groups:
            name = param_group.get('name', 'default')
            lr = param_group['lr']
            wandb_log[f'learning_rate/{name}'] = lr

        accelerator.log(wandb_log, step=global_step)

        # 保存最佳模型 / Save the best model
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            if accelerator.is_local_main_process:
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(accelerator.unwrap_model(model), optimizer,
                              epoch, best_val_loss, save_path, lr_scheduler)

        # 定期保存 / Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0 and accelerator.is_local_main_process:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(accelerator.unwrap_model(model), optimizer, 
                          epoch, best_val_loss, save_path, lr_scheduler)
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()