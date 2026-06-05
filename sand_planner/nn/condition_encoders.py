import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional

from sand_planner.nn.depth_encoder import DepthImageEncoder


class TrajectoryConditionEncoder(nn.Module):
    """轨迹条件编码器。 / Trajectory condition encoder."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, feature_dim: int = 512):
        super().__init__()
        # 输入是 3 维: (x, y, z)
        # Input is 3-dimensional: (x, y, z)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.feature_dim = feature_dim

    def forward(self, end_relative_pose: torch.Tensor) -> torch.Tensor:
        """
        编码轨迹终点的相对坐标。
        Encode the relative coordinates of the trajectory endpoint.

        Args:
            end_relative_pose: (batch_size, 3) - 终点相对坐标 / endpoint relative coordinates.
        Returns:
            encoded: (batch_size, feature_dim) - 编码后的特征 / encoded features.
        """
        return self.encoder(end_relative_pose)


class InitialTurnEncoder(nn.Module):
    """初始转向编码器 - 编码轨迹起始的转向方向 (1 维 Y 分量)。 / Initial turn encoder - encodes the trajectory's initial turn direction (1-D Y component).

    用于提供历史运动信息，帮助模型生成更平滑连续的轨迹。
    输入是归一化后的 (CP[1] + CP[2]) / 2 的 Y 分量。

    Provides historical motion information to help the model generate smoother,
    more continuous trajectories. The input is the normalized Y component of
    (CP[1] + CP[2]) / 2.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # 简单 MLP: 1 维 -> feature_dim
        # Simple MLP: 1-D -> feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, feature_dim)
        )

        # Null token: 表示"没有历史转向信息"(推理时首次规划)
        # Null token: indicates "no historical turn information" (first plan at inference time)
        self.null_token = nn.Parameter(torch.zeros(feature_dim))

        # 小初始化避免破坏已有特征
        # Small initialization to avoid disrupting existing features
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)

        print(f"🔄 InitialTurnEncoder 已创建: 1维转向 → {feature_dim}维特征")

    def forward(self, initial_turn: torch.Tensor, has_initial_turn: torch.Tensor) -> torch.Tensor:
        """
        将初始转向值编码为特征，缺失历史时回退到 null token。
        Encode the initial turn value into features, falling back to the null token when history is missing.

        Args:
            initial_turn: (B,) 或 (B, 1) 初始转向值 (归一化后的 Y 分量) / initial turn value (normalized Y component).
            has_initial_turn: (B,) bool 是否有有效的初始转向 / whether a valid initial turn exists.
        Returns:
            features: (B, feature_dim) 编码后的特征 / encoded features.
        """
        # 确保输入是 (B, 1) 形状
        # Ensure the input has shape (B, 1)
        if initial_turn.dim() == 1:
            initial_turn = initial_turn.unsqueeze(1)  # (B,) -> (B, 1)

        batch_size = initial_turn.shape[0]
        device = initial_turn.device

        # 编码转向 / Encode the turn
        encoded = self.encoder(initial_turn)  # (B, feature_dim)

        # 对于没有历史的样本，使用 null token
        # For samples without history, use the null token
        null_expanded = self.null_token.unsqueeze(0).expand(batch_size, -1)  # (B, feature_dim)

        # 确保 has_initial_turn 是 tensor
        # Ensure has_initial_turn is a tensor
        if isinstance(has_initial_turn, bool):
            has_mask = torch.full((batch_size,), has_initial_turn, dtype=torch.float32, device=device)
        else:
            has_mask = has_initial_turn.float()

        has_mask = has_mask.unsqueeze(1)  # (B, 1)

        output = encoded * has_mask + null_expanded * (1 - has_mask)
        return output  # (B, feature_dim)


class ConcatConditionEncoder(nn.Module):
    """条件编码器 - 拼接深度和轨迹特征后通过 Transformer 处理。 / Condition encoder - concatenates depth and trajectory features, then processes them with a Transformer."""

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
            input_dim=3, feature_dim=feature_dim  # 现在输入 3 维: (x, y, z) / input is now 3-D: (x, y, z)
        )

        # 初始转向编码器 / Initial turn encoder
        if self.use_initial_turn:
            self.initial_turn_encoder = InitialTurnEncoder(feature_dim)
            self.turn_ln = nn.LayerNorm(feature_dim)

        # LayerNorm
        self.depth_ln = nn.LayerNorm(feature_dim)
        self.traj_ln = nn.LayerNorm(feature_dim)

        # 运动 token 的 LayerNorm / LayerNorm for the motion tokens
        if self.use_tea:
            self.motion_ln = nn.LayerNorm(feature_dim)

        # 类型编码 (Type Embedding) - 区分不同模态的 token
        # Type embedding - distinguishes tokens from different modalities.
        # 类型 0: depth tokens (视觉空间特征) / type 0: depth tokens (visual spatial features)
        # 类型 1: motion tokens (运动差分特征) / type 1: motion tokens (motion difference features)
        # 类型 2: initial_turn token (初始转向特征) / type 2: initial_turn token (initial turn feature)
        # 类型 3: trajectory tokens (目标轨迹特征) / type 3: trajectory tokens (target trajectory features)
        if self.use_type_embed:
            num_types = 4 if self.use_initial_turn else 3
            self.type_embed = nn.Embedding(num_types, feature_dim)
            # 关键修复: 小初始化避免破坏已有特征分布
            # Key fix: small initialization to avoid disrupting the existing feature distribution
            nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)
            type_names = "depth/motion/turn/trajectory" if self.use_initial_turn else "depth/motion/trajectory"
            print(f"🎯 Type Embedding 已启用: 区分 {type_names} {num_types}种模态 (小初始化 std=0.02)")

        # 最简单的 Transformer Encoder / The simplest Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 总位移预测头: 预测第 1 帧到第 4 帧的相对位姿变化
        # Total-displacement prediction head: predicts the relative pose change from frame 1 to frame 4
        self.sequence_pose_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 5),  # [dx, dy, dz, dyaw, dpitch] - 总位移(完整 5 维) / total displacement (full 5-D)
            nn.Tanh()
        )

        # 位姿约束范围 / Pose constraint ranges
        self.max_translation = 1.0  # 4 帧之间最大位移 1 米 / max displacement of 1 m across 4 frames
        self.max_rotation = 0.785   # 最大旋转 45 度 / max rotation of 45 degrees

    def forward(self,
                depth_sequence: torch.Tensor,
                end_relative_pose: torch.Tensor,
                initial_turn: Optional[torch.Tensor] = None,
                has_initial_turn: Optional[torch.Tensor] = None,
                cfg_mask: Optional[torch.Tensor] = None,
                return_pose_features: bool = False) -> torch.Tensor:
        """将深度序列空间 tokens 和目标姿态分别编码、LayerNorm 后拼接，然后通过 GPT 式 Transformer 处理。
        Separately encode the depth-sequence spatial tokens and the target pose, apply LayerNorm,
        concatenate them, and process the result with a GPT-style Transformer.

        Args:
            depth_sequence: (B, seq_len, 1, H, W) - 深度图像序列 / depth image sequence.
            end_relative_pose: (B, 3) - 轨迹终点相对坐标 / trajectory endpoint relative coordinates.
            initial_turn: (B,) - 初始转向值(可选) / initial turn value (optional).
            has_initial_turn: (B,) - 是否有有效的初始转向(可选) / whether a valid initial turn exists (optional).
            cfg_mask: (B,) - CFG 掩码，True 表示使用条件，False 表示丢弃条件(用于 CFG 训练) / CFG mask; True keeps the condition, False drops it (for CFG training).
            return_pose_features: bool - 是否返回用于位姿预测的中间特征 / whether to also return the intermediate features used for pose prediction.
        """

        # 编码深度图像序列为空间 tokens + 运动 tokens
        # Encode the depth image sequence into spatial tokens + motion tokens
        depth_tokens, motion_tokens = self.depth_encoder(depth_sequence)
        # depth_tokens: (B, seq_len*N, C) 或 (B, N, C)，取决于 fusion_strategy / depends on fusion_strategy
        # motion_tokens: (B, 3, C) if use_tea else None

        # 编码轨迹终点 / Encode the trajectory endpoint
        traj_features = self.trajectory_encoder(end_relative_pose).unsqueeze(1)  # (B, 1, C)

        # 编码初始转向 / Encode the initial turn
        turn_token = None
        if self.use_initial_turn:
            batch_size_local = depth_tokens.shape[0]
            device_local = depth_tokens.device

            if initial_turn is not None:
                # 有 initial_turn 输入，使用编码器处理
                # initial_turn is provided, process it with the encoder
                turn_features = self.initial_turn_encoder(initial_turn, has_initial_turn)  # (B, C)
            else:
                # 没有 initial_turn 输入(推理时冷启动)，使用 null token
                # No initial_turn input (cold start at inference time), use the null token
                turn_features = self.initial_turn_encoder.null_token.unsqueeze(0).expand(batch_size_local, -1)  # (B, C)

            turn_token = self.turn_ln(turn_features).unsqueeze(1)  # (B, 1, C)

        # 先进行 LayerNorm，再应用 CFG mask
        # Apply LayerNorm first, then the CFG mask
        enhanced_depth_tokens = self.depth_ln(depth_tokens)
        enhanced_traj_features = self.traj_ln(traj_features)  # 先 LayerNorm / LayerNorm first

        # TEA 运动 tokens 的 LayerNorm / LayerNorm for the TEA motion tokens
        if motion_tokens is not None:
            enhanced_motion_tokens = self.motion_ln(motion_tokens)  # (B, 3, C)
        else:
            enhanced_motion_tokens = None

        # 复制目标 token 3 次以增加其权重并兼容多模态架构
        # Replicate the target token 3 times to increase its weight and match the multimodal architecture
        enhanced_traj_features = enhanced_traj_features.repeat(1, 3, 1)  # (B, 1, C) -> (B, 3, C)

        # 添加类型编码 (Type Embedding) / Add the type embedding
        if self.use_type_embed:
            batch_size = enhanced_depth_tokens.shape[0]
            device = enhanced_depth_tokens.device

            # 类型 0: depth tokens / type 0: depth tokens
            num_depth = enhanced_depth_tokens.shape[1]
            type_ids_depth = torch.zeros(batch_size, num_depth, dtype=torch.long, device=device)
            type_embed_depth = self.type_embed(type_ids_depth)  # (B, num_depth, C)
            enhanced_depth_tokens = enhanced_depth_tokens + type_embed_depth

            # 类型 1: motion tokens (如果存在) / type 1: motion tokens (if present)
            if enhanced_motion_tokens is not None:
                type_ids_motion = torch.ones(batch_size, 3, dtype=torch.long, device=device)
                type_embed_motion = self.type_embed(type_ids_motion)  # (B, 3, C)
                enhanced_motion_tokens = enhanced_motion_tokens + type_embed_motion

            # 类型 2: initial_turn token (如果存在) / type 2: initial_turn token (if present)
            if turn_token is not None:
                type_ids_turn = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
                type_embed_turn = self.type_embed(type_ids_turn)  # (B, 1, C)
                turn_token = turn_token + type_embed_turn

            # 类型 3(或 2): trajectory tokens / type 3 (or 2): trajectory tokens
            traj_type_id = 3 if self.use_initial_turn else 2
            type_ids_traj = torch.full((batch_size, 3), traj_type_id, dtype=torch.long, device=device)
            type_embed_traj = self.type_embed(type_ids_traj)  # (B, 3, C)
            enhanced_traj_features = enhanced_traj_features + type_embed_traj

        # CFG 处理: 在 LayerNorm 之后用 mask 置零，避免 LayerNorm 泄漏偏置
        # CFG handling: zero out with the mask after LayerNorm to avoid LayerNorm leaking a bias
        if cfg_mask is not None:
            # cfg_mask 为 True 的样本使用正常条件，False 的样本置零
            # Samples with cfg_mask=True keep their normal conditions; samples with False are zeroed
            batch_size = enhanced_traj_features.shape[0]
            mask_expanded = cfg_mask.view(batch_size, 1, 1)  # (B, 1, 1)

            # 轨迹条件: 根据配置决定是否丢弃 / Trajectory condition: drop it depending on the config
            if self.cfg_drop_trajectory:
                enhanced_traj_features = enhanced_traj_features * mask_expanded.float()

            # 深度条件: 根据配置决定是否丢弃 / Depth condition: drop it depending on the config
            if self.cfg_drop_depth:
                enhanced_depth_tokens = enhanced_depth_tokens * mask_expanded.float()
                # 如果丢弃深度，也丢弃运动信息(因为运动是从深度序列提取的)
                # If depth is dropped, also drop the motion information (motion is extracted from the depth sequence)
                if enhanced_motion_tokens is not None:
                    enhanced_motion_tokens = enhanced_motion_tokens * mask_expanded.float()

        # 拼接经过 LayerNorm 和 CFG 处理的 tokens: [depth, motion, turn, trajectory]
        # Concatenate the LayerNorm- and CFG-processed tokens: [depth, motion, turn, trajectory]
        tokens_to_concat = [enhanced_depth_tokens]
        if enhanced_motion_tokens is not None:
            tokens_to_concat.append(enhanced_motion_tokens)  # 添加 3 个运动 token / append 3 motion tokens
        if turn_token is not None:
            tokens_to_concat.append(turn_token)  # 添加 1 个转向 token / append 1 turn token
        tokens_to_concat.append(enhanced_traj_features)

        combined_tokens = torch.cat(tokens_to_concat, dim=1)
        # 完整版: (B, depth_tokens+3(motion)+1(turn)+3(traj), C)
        # Full version: (B, depth_tokens+3(motion)+1(turn)+3(traj), C)

        # 通过简单的 Transformer Encoder 处理拼接后的 tokens
        # Process the concatenated tokens with the simple Transformer encoder
        transformer_output = self.transformer(combined_tokens)

        # CFG 后处理: Transformer 的 self-attention 会破坏零 token 状态，需要重新应用 mask
        # CFG post-processing: the Transformer's self-attention corrupts the zeroed token state, so re-apply the mask
        if cfg_mask is not None:
            batch_size = transformer_output.shape[0]
            mask_expanded = cfg_mask.view(batch_size, 1, 1)  # (B, 1, 1)

            # 计算轨迹 token 的位置(最后 3 个) / Locate the trajectory tokens (the last 3)
            traj_start_idx = -3

            # 重新对轨迹 token 应用 mask / Re-apply the mask to the trajectory tokens
            if self.cfg_drop_trajectory:
                transformer_output[:, traj_start_idx:, :] = transformer_output[:, traj_start_idx:, :] * mask_expanded.float()

            # 如果也丢弃深度条件 / If the depth condition is also dropped
            if self.cfg_drop_depth:
                # 深度 tokens + 运动 tokens 的范围 / Range covering depth tokens + motion tokens
                depth_motion_end_idx = traj_start_idx
                transformer_output[:, :depth_motion_end_idx, :] = transformer_output[:, :depth_motion_end_idx, :] * mask_expanded.float()

        # 如果需要返回位姿预测特征 / If pose-prediction features are requested
        if return_pose_features:
            # 直接使用 transformer 输出的全局特征(所有 tokens 的平均)
            # 这样可以充分利用 transformer 处理后的信息，包括深度和轨迹的交互
            # Directly use the global feature from the transformer output (mean over all tokens),
            # which fully leverages the post-transformer information, including depth-trajectory interactions.
            pose_features = transformer_output.mean(dim=1)  # (B, C) - 全局平均池化 / global average pooling
            return transformer_output, pose_features

        return transformer_output


class CrossAttnConditionEncoder(nn.Module):
    """条件编码器 - 使用 Cross Attention 将轨迹目标注入到深度特征中。 / Condition encoder - injects the trajectory target into the depth features via cross-attention."""

    def __init__(self,
                 feature_dim: int = 256,
                 seq_len: int = 2,
                 num_transformer_layers: int = 2, # 默认改为 2 层，响应之前的分析 / default changed to 2 layers per earlier analysis
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

        # 类型编码 (Type Embedding) / Type embedding
        if self.use_type_embed:
            self.type_embed = nn.Embedding(3, feature_dim)
            print(f"🎯 Type Embedding 已启用 (CrossAttn): 区分 depth/motion/trajectory")

        # 混合架构: Self-Attention(理解环境) + Cross-Attention(注入目标)
        # Hybrid architecture: self-attention (understand the environment) + cross-attention (inject the target).
        # Stage 1: Self-Attention - 视觉 token 之间的交互，理解障碍物空间结构
        # Stage 1: self-attention - interaction among visual tokens to understand obstacle spatial structure.
        # 关键发现: 3 层 Self-Attn 是感知低障碍物的最小深度
        # Key finding: 3 self-attention layers is the minimum depth for perceiving low obstacles.
        #   - 第 1 层: 理解局部边缘和纹理 / layer 1: understand local edges and textures
        #   - 第 2 层: 理解空间关系和形状 / layer 2: understand spatial relations and shapes
        #   - 第 3 层: 理解地面 vs 障碍物的语义差异 / layer 3: understand the semantic difference between ground and obstacles
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        # 3+1 分割: 75% 给视觉理解，25% 给目标注入
        # 3+1 split: 75% for visual understanding, 25% for target injection
        num_self_layers = max(1, int(num_transformer_layers * 0.75))
        self.self_attention = nn.TransformerEncoder(self_attn_layer, num_layers=num_self_layers)

        # Stage 2: Cross-Attention - 将目标信息注入到已理解的视觉特征中
        # Stage 2: cross-attention - inject target information into the already-understood visual features.
        # 1 层 Cross-Attn 足够: 视觉特征已经充分编码，只需高效注入目标
        # 1 cross-attention layer is enough: the visual features are already well encoded, so only efficient target injection is needed.
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

        # 编码深度图像序列 + 运动 tokens / Encode the depth image sequence + motion tokens
        depth_tokens, motion_tokens = self.depth_encoder(depth_sequence)
        # depth_tokens: (B, N, C)
        # motion_tokens: (B, 3, C) if use_tea else None

        # 编码轨迹终点 (B, 1, C) / Encode the trajectory endpoint (B, 1, C)
        traj_features = self.trajectory_encoder(end_relative_pose).unsqueeze(1)

        # LayerNorm
        enhanced_depth_tokens = self.depth_ln(depth_tokens)
        enhanced_traj_features = self.traj_ln(traj_features)

        # 运动 tokens 的 LayerNorm / LayerNorm for the motion tokens
        if motion_tokens is not None:
            enhanced_motion_tokens = self.motion_ln(motion_tokens)
        else:
            enhanced_motion_tokens = None

        # 添加类型编码 (Type Embedding) / Add the type embedding
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
                # 如果丢弃深度，也丢弃运动信息 / If depth is dropped, also drop the motion information
                if enhanced_motion_tokens is not None:
                    enhanced_motion_tokens = enhanced_motion_tokens * mask_expanded.float()

        # 拼接 depth 和 motion tokens 用于 Self-Attention
        # Concatenate depth and motion tokens for self-attention
        if enhanced_motion_tokens is not None:
            depth_motion_tokens = torch.cat([enhanced_depth_tokens, enhanced_motion_tokens], dim=1)
        else:
            depth_motion_tokens = enhanced_depth_tokens

        # 两阶段处理: / Two-stage processing:
        # Stage 1: Self-Attention - 视觉 token(含运动)理解障碍物空间关系
        # Stage 1: self-attention - visual tokens (including motion) understand obstacle spatial relations
        spatial_features = self.self_attention(depth_motion_tokens)

        # Stage 2: Cross-Attention - 注入目标信息 / Stage 2: cross-attention - inject the target information
        transformer_output = self.cross_attention(
            tgt=spatial_features,              # 已理解空间的视觉特征 / spatially-aware visual features
            memory=enhanced_traj_features      # 目标信息 / target information
        )

        # CFG 后处理 / CFG post-processing
        if cfg_mask is not None:
             batch_size = transformer_output.shape[0]
             mask_expanded = cfg_mask.view(batch_size, 1, 1)
             if self.cfg_drop_depth:
                 transformer_output = transformer_output * mask_expanded.float()

        # 关键修改: 将目标特征显式拼接到输出序列中
        # 这样 UNet 不仅能看到"被目标调制的视觉特征"，还能直接看到"原始目标指令"
        # 这能有效解决轨迹左右摇摆的问题，给模型一个稳定的"指南针"
        # Key change: explicitly concatenate the target features into the output sequence,
        # so the UNet sees not only the "target-modulated visual features" but also the "raw target command".
        # This effectively resolves left-right trajectory wobble, giving the model a stable "compass".
        final_sequence = torch.cat([transformer_output, enhanced_traj_features], dim=1)

        if return_pose_features:
            # 位姿预测只依赖视觉特征(不应该依赖目标)
            # Pose prediction relies only on the visual features (it should not depend on the target)
            pose_features = transformer_output.mean(dim=1)
            return final_sequence, pose_features

        return final_sequence
