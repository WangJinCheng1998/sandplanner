import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from diffusers import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler
from sand_planner.nn.models.unet.unet_1d_condition import UNet1DConditionModel
from sand_planner.nn.condition_encoders import ConcatConditionEncoder

# 可选项：TrajectoryNormalizer 不一定总是可用
# Optional: TrajectoryNormalizer may not always be available
try:
    from sand_planner.utils.normalize import TrajectoryNormalizer
except ImportError:
    TrajectoryNormalizer = None


class BSplineDDPM(nn.Module):
    """用于生成 B-spline 控制点的 DDPM 模型，基于官方 UNet1DConditionModel。 / DDPM model for generating B-spline control points, built on the official UNet1DConditionModel."""

    def __init__(self,
                 condition_encoder: ConcatConditionEncoder,
                 num_train_timesteps: int = 1000,
                 fix_first_cp_zero: bool = False,
                 normalizer = None,
                 cfg_dropout_prob: float = 0.1,
                 trajectory_interpolation: str = 'bspline',
                 prediction_mode: str = 'control_points',
                 num_control_points: int = 8):
        super().__init__()

        self.condition_encoder = condition_encoder
        self.fix_first_cp_zero = fix_first_cp_zero
        # 若提供，则表示在"归一化空间"中训练；推理时自动归一化/反归一化
        # If provided, training happens in normalized space; inference normalizes/denormalizes automatically
        self.normalizer = normalizer
        # CFG 相关参数
        # CFG-related parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        # 轨迹插值方法
        # Trajectory interpolation method
        self.trajectory_interpolation = trajectory_interpolation
        # 预测模式
        # Prediction mode
        self.prediction_mode = prediction_mode
        # B-spline 控制点数量（= UNet 序列长度，训练/推理必须一致）
        # Number of B-spline control points (= UNet sequence length, must match between training and inference)
        self.num_control_points = int(num_control_points)
        # UNet 有 2 次下采样（因子 4），序列长度必须是 4 的倍数，否则 forward 时
        # skip 连接尺寸不匹配。提前 fail-fast，给出可读的报错。
        # UNet has 2 downsampling stages (factor 4), so the sequence length must be a multiple of 4;
        # otherwise skip-connection sizes mismatch in forward. Fail fast with a readable error.
        if self.num_control_points % 4 != 0:
            raise ValueError(
                f"num_control_points={self.num_control_points} 必须是 4 的倍数"
                f"(如 8/12/16/20)。当前 UNet 有 2 次下采样(因子4)，否则会在 forward 时"
                f"报 skip 连接尺寸不匹配。")

        # 根据预测模式初始化
        # Initialize according to the prediction mode
        if self.prediction_mode == 'waypoints':
            print(f"🔧 预测模式: Waypoints (固定0.2m间隔)")
        else:
            print(f"🔧 预测模式: Control Points")
            # 初始化轨迹插值器（当使用 cubic spline 时）
            # Initialize the trajectory interpolator (when using cubic spline)
            if self.trajectory_interpolation == 'cubic_spline':
                from sand_planner.utils.traj_opt import TrajOpt
                self.traj_opt = TrajOpt()
                print(f"🔧 使用 Cubic Spline 插值生成轨迹")
            else:
                print(f"🔧 使用 B-Spline 插值生成轨迹")

        self.unet = UNet1DConditionModel(
            sample_size=self.num_control_points,  # 目标信号长度（= 控制点数量） / target signal length (= number of control points)
            in_channels=3,
            out_channels=3,
            layers_per_block=2,  # 每个 UNet block 使用的 ResNet 层数 / number of ResNet layers per UNet block
            # 使用 3 个 stage，确保总下采样次数为 3 次（2**3=8），与长度 8 对齐。
            # Use 3 stages so the total number of downsamplings is 3 (2**3=8), aligned with length 8.
            # 优化：减小通道数（64->32）以防止过拟合和"绕远路"。
            # 在 Head Dim 修复为 32 后，模型感知力大增，不需要过大的通道数即可工作。
            # 较小的模型倾向于生成更平滑、更直接的轨迹（正则化效果）。
            # Optimization: reduce channel width (64->32) to prevent overfitting and detouring.
            # After fixing Head Dim to 32, the model's perception improves greatly, so large channel
            # widths are no longer needed. Smaller models tend to produce smoother, more direct
            # trajectories (a regularization effect).
            block_out_channels=(64, 128, 256),
            # 保持 Head Dim=32。对应 Head 数量为：32/32=1, 64/32=2, 128/32=4。
            # 这种配置既保证了每个 Head 的表达能力，又限制了总体的复杂度。
            # Keep Head Dim=32. The resulting head counts are: 32/32=1, 64/32=2, 128/32=4.
            # This configuration preserves per-head expressiveness while limiting overall complexity.
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
            # GroupNorm 配置：使用 32 个组以支持 64/128/256 通道数。
            # GroupNorm config: use 32 groups to support 64/128/256 channel widths.
            norm_num_groups=32,  # 各通道数均可被 32 整除 / each channel width is divisible by 32 (64/128/256)
            # 关键修复：显式设置 cross_attention_dim，避免默认值 1280。
            # Key fix: set cross_attention_dim explicitly to avoid the default of 1280.
            encoder_hid_dim=condition_encoder.feature_dim,  # 256
            cross_attention_dim=condition_encoder.feature_dim,  # 256，避免默认 1280 带来的巨大开销 / 256, avoids the huge cost of the default 1280
        )

        # DDPM 调度器
        # DDPM scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,   # 1000 或 2000 起步 / start from 1000 or 2000
            beta_schedule="squaredcos_cap_v2",         # 推荐 / recommended
            prediction_type="v_prediction",                 # 备选/alt: epsilon
            clip_sample=False,
            timestep_spacing="linspace",               # 训练期通常用线性 / training usually uses linear spacing
            rescale_betas_zero_snr=True,               # 推荐打开 / recommended to enable
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
        """
        训练前向传播。 / Training forward pass.

        Args:
            control_points: (batch_size, 8, 3) - GT 控制点 / ground-truth control points
            depth_sequence: (batch_size, seq_len, 1, H, W) - 深度图序列 / depth image sequence
            end_relative_pose: (batch_size, 3) - 轨迹终点相对坐标 / relative coordinates of the trajectory endpoint
            total_displacement: (batch_size, 5) - 第 1 帧到第 4 帧的总位移 [dx, dy, dz, dyaw, dpitch]（可选） / total displacement from frame 1 to frame 4 [dx, dy, dz, dyaw, dpitch] (optional)
            initial_turn: (batch_size,) - 初始转弯值（可选） / initial turn value (optional)
            has_initial_turn: (batch_size,) - 是否存在有效的初始转弯（可选） / whether a valid initial turn exists (optional)
            pose_loss_weight: float - 位姿损失权重 / pose loss weight
        Returns:
            loss_dict: 包含损失信息的字典 / dictionary containing loss information
        """
        batch_size = control_points.size(0)
        device = control_points.device

        # 将控制点重塑为 (batch_size, 3, 8)，即 (batch, channels, length) 格式
        # Reshape control points to (batch_size, 3, 8), i.e. (batch, channels, length) layout
        control_points_reshaped = control_points.transpose(1, 2)  # (batch_size, 3, 8)
        # 盒约束：夹回训练归一化盒（[-1, 1]）
        # Box constraint: clamp back into the training normalization box ([-1, 1])
        control_points_reshaped = control_points_reshaped.clamp(-1.0, 1.0)

        # 若启用，强制干净样本的第一个控制点为 0（保证数据一致性）
        # If enabled, force the first control point of the clean sample to 0 (for data consistency)
        if self.fix_first_cp_zero:
            control_points_reshaped = control_points_reshaped.clone()
            if self.normalizer is not None:
                # 使用归一化空间中的原点值
                # Use the origin value in normalized space
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(control_points_reshaped.device, dtype=control_points_reshaped.dtype)  # (3,)
                control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
            else:
                control_points_reshaped[:, :, 0] = 0.0

        # CFG 训练：随机生成 mask 决定是否丢弃条件
        # CFG training: randomly generate a mask to decide whether to drop the condition
        cfg_mask = None
        if self.training and self.cfg_dropout_prob > 0:
            cfg_mask = torch.rand(batch_size, device=device) > self.cfg_dropout_prob

        # 位姿预测分支（当提供了 GT 时）
        # Pose prediction branch (when ground truth is provided)
        pose_loss = None
        if total_displacement is not None:
            # 使用 return_pose_features 获取中间特征
            # Use return_pose_features to obtain intermediate features
            encoder_hidden_states, pose_features = self.condition_encoder(
                depth_sequence, end_relative_pose,
                initial_turn=initial_turn, has_initial_turn=has_initial_turn,
                cfg_mask=cfg_mask, return_pose_features=True
            )  # encoder_hidden_states: (B, seq_len+1, C), pose_features: (B, C)

            # 通过 MLP 预测位姿
            # Predict the pose via an MLP
            predicted_displacement = self.condition_encoder.sequence_pose_predictor(pose_features)  # (B, 5)

            # 计算位姿损失（Huber Loss，对离群值更鲁棒）。
            # beta=1.0：在归一化空间 [-1, 1] 中，|error|<1 时用 MSE，>1 时用 L1。
            # Compute the pose loss (Huber loss, more robust to outliers).
            # beta=1.0: in normalized space [-1, 1], use MSE when |error|<1 and L1 when >1.
            pose_loss = F.smooth_l1_loss(predicted_displacement, total_displacement, beta=1.0)
        else:
            # 编码条件：直接返回用于 cross-attention 的 token 序列
            # Encode conditions: directly return the token sequence used for cross-attention
            encoder_hidden_states = self.condition_encoder(
                depth_sequence, end_relative_pose,
                initial_turn=initial_turn, has_initial_turn=has_initial_turn,
                cfg_mask=cfg_mask
            )  # (batch_size, seq_len+1, feature_dim)

        # 随机采样时间步
        # Randomly sample timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()

        # 生成随机噪声
        # Generate random noise
        noise = torch.randn_like(control_points_reshaped)

        # 若启用，不对第一个控制点添加噪声（并在损失中屏蔽）
        # If enabled, do not add noise to the first control point (and mask it out in the loss)
        if self.fix_first_cp_zero:
            noise[:, :, 0] = 0.0

        # 添加噪声
        # Add noise
        noisy_control_points = self.scheduler.add_noise(
            control_points_reshaped, noise, timesteps
        )
        if self.fix_first_cp_zero:
            if self.normalizer is not None:
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(noisy_control_points.device, dtype=noisy_control_points.dtype)
                noisy_control_points[:, :, 0] = zero_norm.view(1, 3)
            else:
                noisy_control_points[:, :, 0] = 0.0

        # 预测 v-parametrization：使用官方 UNet 接口，确保与 UNet 的 dtype 对齐
        # Predict the v-parametrization: use the official UNet interface and align with the UNet dtype
        unet_dtype = self.unet.dtype
        noisy_control_points = noisy_control_points.to(dtype=unet_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)
        v_pred = self.unet(
            sample=noisy_control_points,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # 计算 v-parametrization 的目标值。
        # 使用 diffusers 自带的方法计算 v_target，更准确且与库保持一致。
        # Compute the target for the v-parametrization.
        # Use the built-in diffusers method to compute v_target; it is more accurate and consistent with the library.
        v_target = self.scheduler.get_velocity(control_points_reshaped, noise, timesteps)

        # 计算损失
        # Compute the loss
        if self.fix_first_cp_zero:
            # 屏蔽第一个控制点对损失的贡献；保证 dtype 一致，并在 float32 中累加以提升数值稳定性。
            # Mask out the first control point's contribution to the loss; keep dtype consistent and
            # accumulate in float32 for better numerical stability.
            mask = torch.ones_like(v_pred)
            mask[:, :, 0] = 0.0
            diff = (v_pred - v_target.to(dtype=v_pred.dtype)) * mask
            trajectory_loss = (diff.float().pow(2).sum() / mask.float().sum().clamp_min(1.0))
        else:
            trajectory_loss = F.mse_loss(v_pred, v_target.to(dtype=v_pred.dtype))

        # 综合损失：轨迹损失 + 位姿损失
        # Combined loss: trajectory loss + pose loss
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
        """
        支持 CFG 的推理采样。 / Inference sampling with CFG support.

        Args:
            depth_sequence: (batch_size, seq_len, 1, H, W) - 深度图序列 / depth image sequence
            end_relative_pose: (batch_size, 3) - 轨迹终点相对坐标 / relative coordinates of the trajectory endpoint
            num_inference_steps: 推理步数 / number of inference steps
            cfg_scale: CFG 引导强度，1.0 表示无引导，>1.0 表示增强条件引导 / CFG guidance scale; 1.0 means no guidance, >1.0 strengthens conditional guidance
            initial_turn: (batch_size,) 初始转弯值（可选，冷启动时为 None 或 has_initial_turn=False） / initial turn value (optional; None or has_initial_turn=False at cold start)
            has_initial_turn: (batch_size,) 是否存在有效的初始转弯（可选） / whether a valid initial turn exists (optional)
        Returns:
            control_points: (batch_size, 8, 3) - 生成的控制点 / generated control points
        """
        # eval 模式 + 统一 device/dtype
        # eval mode + unified device/dtype
        self.eval()
        batch_size = depth_sequence.size(0)
        device = next(self.parameters()).device
        unet_dtype = self.unet.dtype

        # 将条件移动到同一 device（编码器内部仍按其自身精度工作）
        # Move conditions to the same device (the encoder still works at its own precision internally)
        depth_sequence = depth_sequence.to(device)

        # 编码条件：返回 token 序列 (batch_size, seq_len+1, feature_dim)。
        # 若训练在归一化空间内，则对条件终点做归一化。
        # Encode conditions: return the token sequence (batch_size, seq_len+1, feature_dim).
        # If training is done in normalized space, normalize the conditional endpoint.
        cond_end = end_relative_pose.to(device)
        if self.normalizer is not None:
            end_np = end_relative_pose.detach().cpu().numpy()
            end_norm_np = self.normalizer.normalize(end_np)
            cond_end = torch.from_numpy(end_norm_np).to(device=device, dtype=end_relative_pose.dtype)

        # 处理 initial_turn 参数
        # Handle the initial_turn argument
        cond_initial_turn = None
        cond_has_initial_turn = None
        if initial_turn is not None:
            cond_initial_turn = initial_turn.to(device)
        if has_initial_turn is not None:
            cond_has_initial_turn = has_initial_turn.to(device)
        else:
            # 如果未提供 has_initial_turn，则根据 initial_turn 是否为 None 判断。
            # 推理时：initial_turn=None 表示冷启动。
            # If has_initial_turn is not provided, infer it from whether initial_turn is None.
            # At inference: initial_turn=None indicates a cold start.
            if cond_initial_turn is None:
                cond_has_initial_turn = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                cond_has_initial_turn = torch.ones(batch_size, dtype=torch.bool, device=device)

        # CFG 推理：准备条件编码与无条件编码
        # CFG inference: prepare conditional and unconditional encodings
        use_cfg = cfg_scale > 1.0
        if use_cfg:
            # 条件编码（保留正常条件）
            # Conditional encoding (keep the normal condition)
            encoder_hidden_states_cond = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn,
                cfg_mask=torch.ones(batch_size, device=device, dtype=torch.bool)
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

            # 无条件编码（丢弃条件）
            # Unconditional encoding (drop the condition)
            encoder_hidden_states_uncond = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn,
                cfg_mask=torch.zeros(batch_size, device=device, dtype=torch.bool)
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

            # 合并为一个批次以进行并行推理
            # Concatenate into a single batch for parallel inference
            encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
        else:
            # 标准推理（无 CFG）
            # Standard inference (no CFG)
            encoder_hidden_states = self.condition_encoder(
                depth_sequence, cond_end,
                initial_turn=cond_initial_turn, has_initial_turn=cond_has_initial_turn
            ).to(dtype=unet_dtype)  # (batch_size, seq_len+1, feature_dim)

        # 使用 DPMSolverMultistepScheduler 进行推理（继承训练用的 DDPM 调度器配置）
        # Use DPMSolverMultistepScheduler for inference (inheriting the training DDPM scheduler config)
        dpm_scheduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config,
            algorithm_type="dpmsolver++",
            prediction_type=self.scheduler.config.prediction_type  # 显式继承 prediction_type / explicitly inherit prediction_type
        )

        dpm_scheduler.set_timesteps(num_inference_steps, device=device)

        # 备选方案：在 set_timesteps 阶段启用 Karras sigmas；若当前 diffusers 版本不支持该参数，则自动回退。
        # Alternative: enable Karras sigmas during set_timesteps; fall back automatically if the current diffusers version lacks this argument.
        # karras_enabled = False
        # try:
        #     dpm_scheduler.set_timesteps(num_inference_steps, device=device, use_karras_sigmas=True)

        #     karras_enabled = True
        # except TypeError:
        #     # 旧版本无此参数：尽量设置属性，再使用兼容签名
        #     # Older versions lack this argument: try setting the attribute, then use a compatible signature
        #     karras_enabled = False
        #     try:
        #         dpm_scheduler.use_karras_sigmas = True  # 某些版本暴露为属性 / some versions expose it as an attribute
        #         karras_enabled = True
        #     except Exception:
        #         pass
        #     dpm_scheduler.set_timesteps(num_inference_steps, device=device)
        # if not karras_enabled:
        #     print("[WARN] Karras sigmas may not be enabled in this diffusers version.")
        # # 可选调试打印：输出是否启用以及首尾若干 sigma/timestep 值
        # # Optional debug print: report whether it is enabled and a few leading/trailing sigma/timestep values
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
                # 调试信息打印失败不影响推理
                # A failure when printing debug info must not affect inference
                pass

        # 初始噪声 - (batch_size, 3, num_control_points) 格式
        # Initial noise in (batch_size, 3, num_control_points) layout
        control_points_reshaped = torch.randn(batch_size, 3, self.num_control_points, device=device, dtype=unet_dtype)
        if self.fix_first_cp_zero:
            if self.normalizer is not None:
                zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(device=device, dtype=control_points_reshaped.dtype)
                control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
            else:
                control_points_reshaped[:, :, 0] = 0.0

        # 去噪过程
        # Denoising loop
        for t in tqdm(dpm_scheduler.timesteps, desc="Sampling"):
            if use_cfg:
                # CFG 推理：并行预测条件分支和无条件分支
                # CFG inference: predict the conditional and unconditional branches in parallel
                timesteps = torch.full((batch_size * 2,), t, device=device, dtype=torch.long)

                # 复制输入以进行并行推理
                # Duplicate the input for parallel inference
                model_in_duplicated = torch.cat([control_points_reshaped, control_points_reshaped], dim=0)
                model_in_duplicated = dpm_scheduler.scale_model_input(model_in_duplicated, t)

                # 预测 v-parametrization（并行）
                # Predict the v-parametrization (in parallel)
                v_pred_combined = self.unet(
                    sample=model_in_duplicated,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # 分离条件预测与无条件预测
                # Split the conditional and unconditional predictions
                v_pred_uncond, v_pred_cond = v_pred_combined.chunk(2, dim=0)

                # CFG: v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
            else:
                # 标准推理（无 CFG）
                # Standard inference (no CFG)
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_in = dpm_scheduler.scale_model_input(control_points_reshaped, t)
                v_pred = self.unet(
                    sample=model_in,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            # 去噪步骤
            # Denoising step
            control_points_reshaped = dpm_scheduler.step(
                v_pred, t, control_points_reshaped
            ).prev_sample
            if self.fix_first_cp_zero:
                if self.normalizer is not None:
                    zero_norm = torch.from_numpy(self.normalizer.normalized_zero()).to(device=device, dtype=control_points_reshaped.dtype)
                    control_points_reshaped[:, :, 0] = zero_norm.view(1, 3)
                else:
                    control_points_reshaped[:, :, 0] = 0.0

        # 重塑为控制点形状 (batch_size, 8, 3)
        # Reshape into control-point shape (batch_size, 8, 3)
        control_points = control_points_reshaped.transpose(1, 2)

        # 若训练在归一化空间内，则将输出反归一化回原始尺度
        # If training is in normalized space, denormalize the output back to the original scale
        if self.normalizer is not None:
            # 归一化空间一般在 [-1, 1]（percentile/zscore 裁剪）；推理阶段先夹紧再反归一化可避免外推放大。
            # Normalized space is typically [-1, 1] (percentile/zscore clipping); clamping before denormalizing at inference avoids extrapolation blow-up.
            control_points = control_points.clamp(-1.0, 1.0)
            cp_np = control_points.detach().cpu().numpy()
            cp_denorm_np = self.normalizer.denormalize(cp_np)
            control_points = torch.from_numpy(cp_denorm_np).to(control_points.device, dtype=control_points.dtype)

        return control_points
