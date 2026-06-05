import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import timm
from typing import Dict, Tuple, Optional


class DepthImageEncoder(nn.Module):
    """深度图像编码器，使用 ResNet18 提取空间特征图。/ Depth image encoder that extracts spatial feature maps with ResNet18."""

    def __init__(self, feature_dim: int = 512, fusion_strategy: str = 'concat', use_tea: bool = False):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.use_tea = use_tea

        # 使用 ResNet18 提取空间特征，取 stride=16 的特征层
        # Extract spatial features with ResNet18, using the stride=16 feature level
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=False,
            features_only=True,
            in_chans=1,
            # 改用第 3 层，stride=16 而非 stride=32 / Use stage 3 (stride=16) instead of stride=32
            out_indices=[3],
        )

        # 获取 backbone 的输出维度与空间尺寸 / Probe backbone output channels and spatial size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 168, 224)
            features = self.backbone(dummy_input)
            backbone_out_dim = features[0].shape[1]  # 通道数 / number of channels
            original_h = features[0].shape[2]        # 原始空间高度 / original spatial height
            original_w = features[0].shape[3]        # 原始空间宽度 / original spatial width

        # 用 AdaptiveAvgPool2d 池化到固定网格 8x12=96 tokens/帧
        # Pool to a fixed grid 8x12=96 tokens per frame via AdaptiveAvgPool2d
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 12))  # 固定输出 8x12=96 个 tokens / fixed 8x12=96 tokens output
        self.spatial_h = 8
        self.spatial_w = 12
        self.num_spatial_tokens = self.spatial_h * self.spatial_w  # 96 个 tokens / 96 tokens

        print(f"🔧 空间特征图: {original_h}×{original_w} → 池化到{self.spatial_h}×{self.spatial_w}={self.num_spatial_tokens}tokens")
        print(f"📊 每帧tokens: {original_h * original_w} → {self.num_spatial_tokens}")
        print(f"🔄 多帧融合策略: {self.fusion_strategy}")

        # 1x1 卷积投影到目标维度 / 1x1 conv projection to the target dimension
        self.spatial_proj = nn.Conv2d(backbone_out_dim, feature_dim, kernel_size=1)

        # 时序注意力机制（仅在 attention 模式下使用）/ Temporal attention (only used in 'attention' mode)
        if self.fusion_strategy == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )

        # 2D 正弦位置编码，注册为 buffer 以更好地表示空间位置关系
        # 2D sinusoidal positional encoding, registered as a buffer to encode spatial layout
        pos_embed_2d = self._create_2d_pos_encoding(feature_dim, self.spatial_h, self.spatial_w)
        self.register_buffer('pos_embed_2d', pos_embed_2d)

        # 时间步嵌入（用于多帧序列）/ Timestep embedding (for multi-frame sequences)
        self.time_embed = nn.Embedding(32, feature_dim)  # 最多支持 32 帧 / supports up to 32 frames

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
            # 关键修复：对 motion encoder 做小初始化，避免初期产生过大扰动
            # Key fix: small-init the motion encoder to avoid large perturbations early on
            for module in self.motion_encoder:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print(f"🔥 TEA-lite 已启用: 将生成 3 个运动差分 token (输入=[g_t, Δg_t], 小初始化)")

    def _create_2d_pos_encoding(self, d_model: int, height: int, width: int) -> torch.Tensor:
        """创建 2D 正弦位置编码。/ Create a 2D sinusoidal positional encoding."""
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D positional encoding"

        # 创建位置编码矩阵 / Allocate the positional encoding matrix
        pos_encoding = torch.zeros(height, width, d_model)

        # 计算频率项 / Compute the frequency terms
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
        """创建 1D 正弦位置编码。/ Create a 1D sinusoidal positional encoding."""
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, depth_images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播：将深度图序列编码为 token 特征。/ Forward pass: encode a depth image sequence into token features.

        Args:
            depth_images: 深度图序列，形状 (batch_size, seq_len, 1, H, W)。/ Depth image sequence of shape (batch_size, seq_len, 1, H, W).
        Returns:
            features: 输出 token 特征，形状 (batch_size, seq_len * num_spatial_tokens, feature_dim)。/ Output token features of shape (batch_size, seq_len * num_spatial_tokens, feature_dim).
            motion_tokens: 启用 TEA 时为 (batch_size, 3, feature_dim) 的 3 个帧间差分 token，否则为 None。/ Three inter-frame difference tokens of shape (batch_size, 3, feature_dim) when use_tea is set, otherwise None.
        """
        batch_size, seq_len, channels, height, width = depth_images.shape

        # 重塑为 (batch_size * seq_len, channels, H, W) / Reshape to (batch_size * seq_len, channels, H, W)
        depth_images = depth_images.reshape(batch_size * seq_len, channels, height, width)

        # 通过 ResNet 提取空间特征图 / Extract spatial feature maps via ResNet
        features = self.backbone(depth_images)[0]  # (batch_size * seq_len, C, H', W')

        # 1x1 卷积投影维度 / 1x1 conv projection of the channel dimension
        features = self.spatial_proj(features)  # (batch_size * seq_len, feature_dim, H', W')

        # 自适应池化到固定网格 8x12=96 tokens / Adaptive pooling to the fixed 8x12=96 token grid
        features = self.adaptive_pool(features)  # (batch_size * seq_len, feature_dim, 8, 12)

        # 展平空间维度为 tokens / Flatten spatial dimensions into tokens
        spatial_features = features.flatten(2).transpose(1, 2)  # (batch_size * seq_len, 96, feature_dim)

        # 添加 2D 位置编码（作为 buffer，会自动跟随模型设备）
        # Add the 2D positional encoding (a buffer that automatically follows the model device)
        spatial_features = spatial_features + self.pos_embed_2d.to(spatial_features.dtype)

        # 重塑回序列维度 / Reshape back to the sequence layout
        spatial_features = spatial_features.reshape(
            batch_size, seq_len, self.num_spatial_tokens, self.feature_dim
        )  # (batch_size, seq_len, num_spatial_tokens, feature_dim)

        # TEA-lite: 在添加时间步嵌入之前提取运动信息（关键修复）
        # TEA-lite: extract motion information before adding the timestep embedding (key fix)
        motion_tokens = None
        if self.use_tea and seq_len > 1:
            # 1. 提取每帧的全局特征向量（空间平均池化）
            #    关键：在尚未叠加时间步嵌入的纯视觉特征上计算
            # 1. Extract the global feature vector per frame (spatial average pooling).
            #    Key point: compute on pure visual features before the timestep embedding is added.
            global_feat_raw = spatial_features.mean(dim=2)  # (B, seq_len, feature_dim)

            # 2. 计算帧间差分：4 帧 → 3 个差分向量
            #    delta[0] = frame[1] - frame[0]（第 1→2 帧的运动）
            #    delta[1] = frame[2] - frame[1]（第 2→3 帧的运动）
            #    delta[2] = frame[3] - frame[2]（第 3→4 帧的运动）
            # 2. Compute inter-frame differences: 4 frames -> 3 difference vectors.
            #    delta[0] = frame[1] - frame[0] (motion from frame 1 to 2)
            #    delta[1] = frame[2] - frame[1] (motion from frame 2 to 3)
            #    delta[2] = frame[3] - frame[2] (motion from frame 3 to 4)
            delta_feat = global_feat_raw[:, 1:] - global_feat_raw[:, :-1]  # (B, seq_len-1, feature_dim)

            # 3. Motion encoder 输入：拼接 [当前帧特征, 帧间差分]，同时保留绝对位置信息(g_t)与相对运动信息(Δg_t)
            # 3. Motion encoder input: concat [current frame feature, inter-frame diff], keeping both
            #    absolute position info (g_t) and relative motion info (Δg_t).
            current_frames = global_feat_raw[:, 1:]  # (B, seq_len-1, feature_dim) - 取后 3 帧 / take the last 3 frames
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
            # 原始拼接策略或单帧 / Plain concatenation strategy, or the single-frame case
            output_features = spatial_features.reshape(
                batch_size, seq_len * self.num_spatial_tokens, self.feature_dim
            )  # (batch_size, seq_len * num_spatial_tokens, feature_dim)
        elif self.fusion_strategy == 'average':
            # 时序平均融合：将多帧平均为单帧的 token 数量
            # Temporal average fusion: average the frames down to a single frame's token count
            output_features = spatial_features.mean(dim=1)  # (batch_size, num_spatial_tokens, feature_dim)
        elif self.fusion_strategy == 'attention':
            # 时序注意力融合 / Temporal attention fusion
            # 重塑为 (batch_size * num_spatial_tokens, seq_len, feature_dim)
            # Reshape to (batch_size * num_spatial_tokens, seq_len, feature_dim)
            spatial_for_attn = spatial_features.transpose(1, 2).reshape(
                batch_size * self.num_spatial_tokens, seq_len, self.feature_dim
            )

            # 应用注意力：融合每个空间位置上的时序信息
            # Apply attention: fuse temporal information at each spatial location
            attn_output, _ = self.temporal_attention(
                spatial_for_attn, spatial_for_attn, spatial_for_attn
            )  # (batch_size * num_spatial_tokens, seq_len, feature_dim)

            # 时序维度平均池化 / Average pooling over the temporal dimension
            attn_pooled = attn_output.mean(dim=1)  # (batch_size * num_spatial_tokens, feature_dim)

            # 重塑回 (batch_size, num_spatial_tokens, feature_dim) / Reshape back to (batch_size, num_spatial_tokens, feature_dim)
            output_features = attn_pooled.reshape(
                batch_size, self.num_spatial_tokens, self.feature_dim
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return output_features, motion_tokens
