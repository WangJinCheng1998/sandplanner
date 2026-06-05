# 改编自 HuggingFace diffusers / Adapted from HuggingFace diffusers
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/downsampling.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.normalization import RMSNorm
from .upsampling import upfirdn1d_native

class Downsample1D(nn.Module):
    """带可选卷积的一维下采样层 / A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            输入与输出的通道数 / number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            是否使用卷积 / option to use a convolution.
        out_channels (`int`, optional):
            输出通道数，默认与 `channels` 相同 / number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            卷积的填充量 / padding for the convolution.
        name (`str`, default `conv`):
            该一维下采样层的名称 / name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)

# 新增函数 / New function
def downsample_1d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""使用给定滤波器对一批一维序列进行下采样 / Downsample1D a batch of 1D seqs with the given filter.

    接受形状为 `[N, C, L]` 的一批一维序列，并使用给定滤波器对每条序列进行下采样。
    滤波器会被归一化，使得当输入为常量时，输出按指定的 `gain` 缩放。序列之外的样本
    被视为零，且滤波器以零填充，使其形状为下采样因子的整数倍。
    Accepts a batch of 1D seqs of the shape `[N, C, L]` and downsamples each seq with the
    given filter. The filter is normalized so that if the inputs are constant, they will be scaled by the
    specified `gain`. Samples outside the seq are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`torch.Tensor`)
            形状为 `[N, C, L]` 或 `[N, L, C]` 的输入张量 / Input tensor of the shape `[N, C, L]` or `[N, L, C]`.
        kernel (`torch.Tensor`, *optional*):
            形状为 `[firH, firW]` 或 `[firN]`（可分离）的 FIR 滤波器，默认 `[1] * factor`，对应平均池化 /
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            整数下采样因子 / Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            信号幅度的缩放因子 / Scaling factor for signal magnitude.

    Returns:
        output (`torch.Tensor`):
            形状为 `[N, C, L // factor]` 的张量 / Tensor of the shape `[N, C, L // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    assert kernel.ndim == 1
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn1d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output