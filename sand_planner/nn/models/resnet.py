# 改编自 HuggingFace diffusers / Adapted from HuggingFace diffusers
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/resnet.py

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.activations import get_activation
from .downsampling import Downsample1D, downsample_1d
from diffusers.models.normalization import AdaGroupNorm
from .upsampling import Upsample1D, upsample_1d

class ResnetBlock1D(nn.Module):
    r"""
    一维 ResNet 残差块。 / A 1D Resnet block.

    Parameters:
        in_channels (`int`): 输入通道数。 / The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            第一个 conv1d 层的输出通道数；若为 None，则与 `in_channels` 相同。 /
            The number of output channels for the first conv1d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): 使用的 dropout 概率。 / The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): 时间步嵌入的通道数。 / the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): 第一个归一化层使用的分组数。 / The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            第二个归一化层使用的分组数；若为 None，则与 `groups` 相同。 /
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): 归一化使用的 epsilon。 / The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): 使用的激活函数。 / the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): 时间尺度偏移配置。 / Time scale shift config.
            默认对时间步嵌入采用简单的偏移机制进行条件注入；选择 "scale_shift" 可获得带尺度与偏移的更强条件注入。 /
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        kernel (`torch.Tensor`, optional, default to None): FIR 滤波器。 / FIR filter
        output_scale_factor (`float`, *optional*, default to be `1.0`): 输出使用的缩放因子。 / the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            若为 `True`，在跳连接上添加一个 1x1 的 nn.conv1d 层。 /
            If `True`, add a 1x1 nn.conv1d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): 若为 `True`，添加一个上采样层。 / If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): 若为 `True`，添加一个下采样层。 / If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  若为 `True`，为 `conv_shortcut` 输出添加可学习的偏置。 /
            If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_1d_out_channels (`int`, *optional*, default to `None`): 输出通道数；若为 None，则与 `out_channels` 相同。 /
            the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # 可选值/options: default, scale_shift
        kernel: Optional[torch.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_1d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm1D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm1D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_1d_out_channels = conv_1d_out_channels or out_channels
        self.conv2 = nn.Conv1d(out_channels, conv_1d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_1d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample1D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_1d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool1d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample1D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_1d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv1d(
                in_channels,
                conv_1d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc 在较大批量下会失败，参见 https://github.com/huggingface/diffusers/issues/984
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        
        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# 源自 unet_rl.py / from unet_rl.py
def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")