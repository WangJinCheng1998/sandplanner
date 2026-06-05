# 改编自 HuggingFace diffusers
# Adapted from HuggingFace diffusers
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/upsampling.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.normalization import RMSNorm

# 新增 / New
class Upsample1D(nn.Module):
    """带可选卷积的一维上采样层。 / A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            输入与输出的通道数。 / number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            是否使用卷积。 / option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            是否使用转置卷积。 / option to use a convolution transpose.
        out_channels (`int`, optional):
            输出通道数，默认与 `channels` 相同。 / number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            一维上采样层的名称。 / name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        if output_size is None:
            outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        else:
            outputs = F.interpolate(inputs, size=output_size, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs

# 新增 / New
def upfirdn1d_native(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    pad: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    pad_0 = pad[0]
    pad_1 = pad[1]

    _, channel, in_len = tensor.shape
    # 形状 / shape: [B*C, L, 1]
    tensor = tensor.reshape(-1, in_len, 1)

    _, in_len, minor = tensor.shape
    kernel_len = len(kernel)

    # 形状 / shape: [B*C, L, 1, 1]
    out = tensor.view(-1, in_len, 1, minor)
    # 在样本之间插入 (up - 1) 个零，形状 / insert (up - 1) zeros between samples, shape: [B*C, L, u, 1]
    out = F.pad(out, [0, 0, 0, up - 1])
    # 形状 / shape: [B*C, uL, 1]
    out = out.view(-1, in_len * up, minor)

    out = F.pad(out, [0, 0, max(pad_0, 0), max(pad_1, 0)])
    # 必要时移回 mps 设备 / Move back to mps if necessary
    out = out.to(tensor.device)
    # 形状 / shape: [B*C, uL(padded), 1]
    out = out[:, max(-pad_0, 0) : out.shape[1] - max(-pad_1, 0), :]

    # 形状 / shape: [B*C, 1, uL(padded)]
    out = out.permute(0, 2, 1)
    out = out.reshape([-1, 1, in_len * up + pad_0 + pad_1])
    # 翻转后的 FIR 卷积核，形状 / flipped FIR kernel, shape: [1, 1, K]
    w = torch.flip(kernel, [0,]).view(1, 1, kernel_len)
    out = F.conv1d(out, w)
    out = out.reshape(-1, minor, in_len * up + pad_0 + pad_1 - kernel_len + 1)
    # 形状 / shape: [B*C, uL(conved), 1]
    out = out.permute(0, 2, 1)
    # 按 down 进行下采样，形状 / downsample by stride down, shape: [B*C, out_len, 1]
    out = out[:, ::down, :]

    out_len = (in_len * up + pad_0 + pad_1 - kernel_len) // down + 1

    return out.view(-1, channel, out_len)

# 新增 / New
def upsample_1d(
    hidden_states: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> torch.Tensor:
    r"""用给定滤波器对一批一维序列进行上采样。 / Upsample1D a batch of 1D seqs with the given filter.

    接收形状为 `[N, C, L]` 的一批一维序列（借助 reshape 方法），并用给定滤波器对每条序列做上采样。
    滤波器会被归一化，使得当输入为常量时，输出按指定的 `gain` 缩放。序列范围之外的样本视为零，
    并对滤波器补零，使其形状为上采样因子的整数倍。

    Accepts a batch of 1D seqs of the shape `[N, C, L]` (thanks to reshape method) and upsamples each seq with the
    given filter. The filter is normalized so that if the inputs are constant, they will be scaled by the
    specified `gain`. Samples outside the seq are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the upsampling factor.

    Args:
        hidden_states (`torch.Tensor`):
            形状为 `[N, C, L]` 或 `[N, L, C]` 的输入张量。 / Input tensor of the shape `[N, C, L]` or `[N, L, C]`.
        kernel (`torch.Tensor`, *optional*):
            形状为 `[firH, firW]` 或 `[firN]`（可分离）的 FIR 滤波器；默认为 `[1] * factor`，对应最近邻上采样。
            / FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            整数上采样因子。 / Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            信号幅度的缩放因子（默认 1.0）。 / Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`torch.Tensor`):
            形状为 `[N, C, L * factor]` 的张量。 / Tensor of the shape `[N, C, L * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    assert kernel.ndim == 1
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * factor)
    pad_value = kernel.shape[0] - factor
    output = upfirdn1d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output