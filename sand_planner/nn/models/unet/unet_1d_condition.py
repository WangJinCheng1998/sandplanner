# 改编自 HuggingFace diffusers（从 2D 适配为 1D）。
# Adapted from HuggingFace diffusers (ported from 2D to 1D).
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/unets/unet_2d_condition.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
# LoRA 相关 mixin；此处沿用 2D 版本，1D 是否完全适用仍待确认。
# LoRA-related mixins; reused from the 2D version, full applicability to 1D is TBD.
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from diffusers.models.embeddings import GaussianFourierProjection, TextTimeEmbedding, TimestepEmbedding, Timesteps
from ..embeddings import SeqProjection, SeqTimeEmbedding, TextSeqProjection, TextSeqTimeEmbedding
from diffusers.models.modeling_utils import ModelMixin
from .unet_1d_blocks import (
    get_down_block,
    get_mid_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name  # 模块日志器 / module-level logger


@dataclass
class UNet1DConditionOutput(BaseOutput):
    """
    [`UNet1DConditionModel`] 的输出。/ The output of [`UNet1DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, length)`):
            以 `encoder_hidden_states` 为条件得到的隐状态输出，即模型最后一层的输出。/
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None


class UNet1DConditionModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    r"""
    条件式 1D UNet 模型：接收带噪样本、条件状态和一个时间步，返回与样本同形状的输出。/
    A conditional 1D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    本模型继承自 [`ModelMixin`]。通用方法（如下载、保存）请参阅父类文档。/
    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*, defaults to `None`):
            输入/输出样本的长度。/ length of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): 输入样本的通道数。/ Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): 输出的通道数。/ Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): 是否对输入样本做中心化。/ Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            时间嵌入中是否将 sin 与 cos 互换。/ Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): 应用于时间嵌入的频率偏移。/ The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock1D", "CrossAttnDownBlock1D", "CrossAttnDownBlock1D", "DownBlock1D")`):
            使用的下采样块类型元组。/ The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1DCrossAttn"`):
            UNet 中间块的类型，可取 `UNetMidBlock1DCrossAttn`、`UNetMidBlock1D` 或
            `UNetMidBlock1DSimpleCrossAttn`；若为 `None` 则跳过中间块。/
            Block type for middle of UNet, it can be one of `UNetMidBlock1DCrossAttn`, `UNetMidBlock1D`, or
            `UNetMidBlock1DSimpleCrossAttn`. If `None`, the mid block layer is skipped.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock1D", "CrossAttnUpBlock1D", "CrossAttnUpBlock1D", "CrossAttnUpBlock1D")`):
            使用的上采样块类型元组。/ The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            基础 transformer 块中是否包含自注意力，参见
            [`~models.attention.BasicTransformerBlock`]。/
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            每个块的输出通道数元组。/ The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): 每个块的层数。/ The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): 下采样卷积使用的 padding。/ The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): 中间块使用的缩放因子。/ The scale factor to use for the mid block.
        dropout (`float`, *optional*, defaults to 0.0): 使用的 dropout 概率。/ The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): 使用的激活函数。/ The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): 归一化使用的分组数；
            若为 `None`，则在后处理中跳过归一化与激活层。/ The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        norm_eps (`float`, *optional*, defaults to 1e-5): 归一化使用的 epsilon。/ The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            交叉注意力特征的维度。/ The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            类型为 [`~models.attention.BasicTransformerBlock`] 的 transformer 块数量。仅对
            [`~models.unets.unet_1d_blocks.CrossAttnDownBlock1D`]、[`~models.unets.unet_1d_blocks.CrossAttnUpBlock1D`]、
            [`~models.unets.unet_1d_blocks.UNetMidBlock1DCrossAttn`] 有效。/
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_1d_blocks.CrossAttnDownBlock1D`], [`~models.unets.unet_1d_blocks.CrossAttnUpBlock1D`],
            [`~models.unets.unet_1d_blocks.UNetMidBlock1DCrossAttn`].
        reverse_transformer_layers_per_block : (`Tuple[Tuple]`, *optional*, defaults to None):
            U-Net 上采样块中类型为 [`~models.attention.BasicTransformerBlock`] 的 transformer 块数量。
            仅当 `transformer_layers_per_block` 为 `Tuple[Tuple]` 类型，且用于
            [`~models.unets.unet_1d_blocks.CrossAttnDownBlock1D`]、[`~models.unets.unet_1d_blocks.CrossAttnUpBlock1D`]、
            [`~models.unets.unet_1d_blocks.UNetMidBlock1DCrossAttn`] 时有效。/
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`], in the upsampling
            blocks of the U-Net. Only relevant if `transformer_layers_per_block` is of type `Tuple[Tuple]` and for
            [`~models.unets.unet_1d_blocks.CrossAttnDownBlock1D`], [`~models.unets.unet_1d_blocks.CrossAttnUpBlock1D`],
            [`~models.unets.unet_1d_blocks.UNetMidBlock1DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            若定义了 `encoder_hid_dim_type`，则 `encoder_hidden_states` 会从 `encoder_hid_dim`
            维度投影到 `cross_attention_dim`。/
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            若给定，则根据 `encoder_hid_dim_type`，将 `encoder_hidden_states` 及可能的其他嵌入
            降投影到维度为 `cross_attention` 的文本嵌入。/
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`int`, *optional*, defaults to 8): 注意力头的维度。/ The dimension of the attention heads.
        num_attention_heads (`int`, *optional*):
            注意力头的数量；若未定义，则默认为 `attention_head_dim`。/
            The number of attention heads. If not defined, defaults to `attention_head_dim`
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): ResNet 块的时间尺度偏移配置
            （参见 [`~models.resnet.ResnetBlock1D`]），可选 `default` 或 `scale_shift`。/ Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock1D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            使用的类别嵌入类型，最终会与时间嵌入相加。可选 `None`、
            `"timestep"`、`"identity"`、`"projection"` 或 `"simple_projection"`。/
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            配置一个可选嵌入，将与时间嵌入相加。可选 `None` 或 "text"；
            "text" 会使用 `TextTimeEmbedding` 层。/
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        addition_time_embed_dim: (`int`, *optional*, defaults to `None`):
            时间步嵌入的维度。/ Dimension for the timestep embeddings.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            当 `class_embed_type` 为 `None` 进行类别条件控制时，需投影到 `time_embed_dim`
            的可学习嵌入矩阵的输入维度。/
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, defaults to `positional`):
            时间步使用的位置嵌入类型，可选 `positional` 或 `fourier`。/
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            投影后时间嵌入维度的可选覆盖值。/ An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, defaults to `None`):
            可选激活函数，仅在时间嵌入传入 UNet 其余部分之前对其使用一次。
            可选 `silu`、`mish`、`gelu` 和 `swish`。/
            Optional activation function to use only once on the time embeddings before they are passed to the rest of
            the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str`, *optional*, defaults to `None`):
            时间步嵌入中使用的第二个激活函数，可选 `silu`、`mish` 和 `gelu`。/
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, defaults to `None`):
            时间步嵌入中 `cond_proj` 层的维度。/ The dimension of `cond_proj` layer in the timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): `conv_in` 层的卷积核大小。/ The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): `conv_out` 层的卷积核大小。/ The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): 当 `class_embed_type="projection"` 时
            `class_labels` 输入的维度；该模式下为必填。/ The dimension of the `class_labels` input when
            `class_embed_type="projection"`. Required when `class_embed_type="projection"`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): 是否将时间嵌入
            与类别嵌入拼接。/ Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            使用 `UNetMidBlock1DSimpleCrossAttn` 时中间块是否使用交叉注意力。若
            `only_cross_attention` 以单个布尔值给出且 `mid_block_only_cross_attention` 为 `None`，
            则将 `only_cross_attention` 的值用作 `mid_block_only_cross_attention`；否则默认为 `False`。/
            Whether to use cross attention with the mid block when using the `UNetMidBlock1DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is `None`, the
            `only_cross_attention` value is used as the value for `mid_block_only_cross_attention`. Default to `False`
            otherwise.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock1D", "CrossAttnUpBlock1D"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = ("CrossAttnDownBlock1D", "CrossAttnDownBlock1D", "CrossAttnDownBlock1D", "DownBlock1D"),
        mid_block_type: Optional[str] = "UNetMidBlock1DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock1D", "CrossAttnUpBlock1D", "CrossAttnUpBlock1D", "CrossAttnUpBlock1D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__()

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # 若未定义 `num_attention_heads`（大多数模型如此），则默认取 `attention_head_dim`。
        # 这一行初看很奇怪，确实如此：其原因是为兼容本库创建之初引入的命名错误，
        # 该错误命名直到很久之后才被发现（见 https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131）。
        # 由于为 40000+ 份配置把 `attention_head_dim` 改名为 `num_attention_heads` 会严重破坏向后兼容，
        # 因此在此处对命名做修正。
        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # 检查输入配置 / Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        # 输入 / input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv1d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # 时间嵌入 / time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

        # 类别嵌入 / class embedding
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # 时间嵌入与类别嵌入拼接，因此传入下采样、中间和上采样块的时间嵌入维度
            # 是常规时间嵌入维度的两倍。
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # 下采样路径 / down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # 中间块 / mid
        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )

        # 统计有多少层会对序列做上采样 / count how many layers upsample the seqs
        self.num_upsamplers = 0

        # 上采样路径 / up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # 除最后一层外，其余层都添加上采样块 / add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # 输出 / out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv1d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    def _check_config(
        self,
        down_block_types: Tuple[str],
        up_block_types: Tuple[str],
        only_cross_attention: Union[bool, Tuple[bool]],
        block_out_channels: Tuple[int],
        layers_per_block: Union[int, Tuple[int]],
        cross_attention_dim: Union[int, Tuple[int]],
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
        reverse_transformer_layers_per_block: bool,
        attention_head_dim: int,
        num_attention_heads: Optional[Union[int, Tuple[int]]],
    ):
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_seq_proj":
            # seq_embed_dim 并不必须等于 `cross_attention_dim`。为避免使 __init__ 过于杂乱，
            # 此处将其设为 `cross_attention_dim`，因为这正是当前唯一用例
            # （`addition_embed_type == "text_seq_proj"`，Kandinsky 2.1）所需的维度。
            # seq_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_seq_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextSeqProjection(
                text_embed_dim=encoder_hid_dim,
                seq_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "seq_proj":
            # Kandinsky 2.2 风格 / Kandinsky 2.2
            self.encoder_hid_proj = SeqProjection(
                seq_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_seq_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # "projection" 类型的 `class_embed_type` 与 "timestep" 类型基本相同，区别在于：
            # 1. `class_labels` 输入不会先转换为正弦嵌入；
            # 2. 它从任意输入维度进行投影。
            #
            # 注意 `TimestepEmbedding` 相当通用，主要由线性层和激活函数构成。用于嵌入真实时间步时，
            # 时间步会先转换为正弦嵌入；因此 `TimestepEmbedding` 也可接收任意向量。
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_seq":
            # text_embed_dim 和 seq_embed_dim 并不必须等于 `cross_attention_dim`。为避免使 __init__ 过于杂乱，
            # 此处将其设为 `cross_attention_dim`，因为这正是当前唯一用例
            # （`addition_embed_type == "text_seq"`，Kandinsky 2.1）所需的维度。
            # text_embed_dim and seq_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_seq"` (Kandinsky 2.1)`
            self.add_embedding = TextSeqTimeEmbedding(
                text_embed_dim=cross_attention_dim, seq_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "seq":
            # Kandinsky 2.2 风格 / Kandinsky 2.2
            self.add_embedding = SeqTimeEmbedding(seq_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_seq'.")

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            注意力处理器的 `dict`：包含模型中所有注意力处理器，以其权重名作为索引。/
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 递归收集 / set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的注意力处理器。/ Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                已实例化的处理器类，或处理器类字典，将作为**所有** `Attention` 层的处理器。

                若 `processor` 为字典，则键需指定到对应交叉注意力处理器的路径。
                在设置可训练的注意力处理器时强烈建议使用字典形式。/
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置为默认的注意力实现。/
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def set_attention_slice(self, slice_size: Union[str, int, List[int]] = "auto"):
        r"""
        启用切片注意力计算。/ Enable sliced attention computation.

        启用该选项后，注意力模块会将输入张量切分为若干片，分多步计算注意力。
        这有助于节省部分显存，代价是速度略有下降。/
        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                取 `"auto"` 时，注意力头的输入减半，分两步计算注意力；取 `"max"` 时，
                每次只运行一个切片以最大限度节省显存；若提供数字，则使用
                `attention_head_dim // slice_size` 个切片，此时 `attention_head_dim`
                必须是 `slice_size` 的整数倍。/
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # 获取注意力层的数量 / retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # 取注意力头尺寸的一半，通常是速度与显存之间较好的折中。
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # 使用尽可能小的切片 / make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # 递归遍历所有子模块；任何暴露了 set_attention_slice 方法的子模块都会收到通知。
        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # TODO: 1D 版本的 FreeU 尚待实现。/ 1D FreeU is yet to be implemented.

    def fuse_qkv_projections(self):
        """
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即 query、key、value）会被融合；
        对于交叉注意力模块，仅融合 key 和 value 投影矩阵。/
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        该 API 为实验性功能。/ This API is experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """若已启用融合的 QKV 投影，则将其禁用。/ Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        该 API 为实验性功能。/ This API is experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: 这里需要 CPU 与 GPU 之间同步，因此尽量直接以张量形式传入时间步。
            # 此处适合用 `match` 语句（Python 3.10+）。/
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # 以兼容 ONNX/Core ML 的方式广播到 batch 维度。
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` 不含任何权重，始终返回 f32 张量；但 time_embedding 实际可能以 fp16 运行，
        # 因此这里需要做类型转换。或许有更好的封装方式。
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` 不含任何权重，始终返回 f32 张量；或许有更好的封装方式。
                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        aug_emb = None
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_seq":
            # Kandinsky 2.1 风格 / Kandinsky 2.1 - style
            if "seq_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_seq' which requires the keyword argument `seq_embeds` to be passed in `added_cond_kwargs`"
                )

            seq_embs = added_cond_kwargs.get("seq_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, seq_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL 风格 / SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "seq":
            # Kandinsky 2.2 风格 / Kandinsky 2.2 - style
            if "seq_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'seq' which requires the keyword argument `seq_embeds` to be passed in `added_cond_kwargs`"
                )
            seq_embs = added_cond_kwargs.get("seq_embeds")
            aug_emb = self.add_embedding(seq_embs)
        return aug_emb

    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_seq_proj":
            # Kandinsky 2.1 风格 / Kandinsky 2.1 - style
            if "seq_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_seq_proj' which requires the keyword argument `seq_embeds` to be passed in  `added_conditions`"
                )

            seq_embeds = added_cond_kwargs.get("seq_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, seq_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "seq_proj":
            # Kandinsky 2.2 风格 / Kandinsky 2.2 - style
            if "seq_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'seq_proj' which requires the keyword argument `seq_embeds` to be passed in  `added_conditions`"
                )
            seq_embeds = added_cond_kwargs.get("seq_embeds")
            encoder_hidden_states = self.encoder_hid_proj(seq_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_seq_proj":
            if "seq_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_seq_proj' which requires the keyword argument `seq_embeds` to be passed in  `added_conditions`"
                )

            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            seq_embeds = added_cond_kwargs.get("seq_embeds")
            seq_embeds = self.encoder_hid_proj(seq_embeds)
            encoder_hidden_states = (encoder_hidden_states, seq_embeds)
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet1DConditionOutput, Tuple]:
        r"""
        [`UNet1DConditionModel`] 的前向方法。/ The [`UNet1DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                带噪输入张量，形状为 `(batch, channel, length)`。/
                The noisy input tensor with the following shape `(batch, channel, length)`.
            timestep (`torch.Tensor` or `float` or `int`): 用于对输入去噪的时间步数。/ The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                编码器隐状态，形状为 `(batch, sequence_length, feature_dim)`。/
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                用于条件控制的可选类别标签，其嵌入会与时间步嵌入相加。/
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                时间步的条件嵌入。若提供，则会与经过 `self.time_embedding` 层的样本相加以得到时间步嵌入。/
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                形状为 `(batch, key_tokens)` 的注意力掩码，作用于 `encoder_hidden_states`。为 `1` 时保留，
                为 `0` 时丢弃。掩码会被转换为偏置，对应"丢弃" token 的注意力分数会加上很大的负值。/
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                若指定，则该 kwargs 字典会传递给在
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
                中 `self.processor` 定义的 `AttentionProcessor`。/
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                包含额外嵌入的 kwargs 字典；若指定，这些嵌入会被加到传入 UNet 各块的嵌入上。/
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                张量元组；若指定，会被加到下采样 UNet 块的残差上。/
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                张量；若指定，会被加到中间 UNet 块的残差上。/
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                需在 UNet 下采样块内部添加的额外残差，例如来自 T2I-Adapter 旁路模型。/
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                形状为 `(batch, sequence_length)` 的交叉注意力掩码，作用于 `encoder_hidden_states`。为 `True` 时保留，
                为 `False` 时丢弃。掩码会被转换为偏置，对应"丢弃" token 的注意力分数会加上很大的负值。/
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`~models.unets.unet_1d_condition.UNet1DConditionOutput`] 而非普通元组。/
                Whether or not to return a [`~models.unets.unet_1d_condition.UNet1DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_1d_condition.UNet1DConditionOutput`] or `tuple`:
                若 `return_dict` 为 True，返回 [`~models.unets.unet_1d_condition.UNet1DConditionOutput`]；
                否则返回元组，其首个元素为样本张量。/
                If `return_dict` is True, an [`~models.unets.unet_1d_condition.UNet1DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 默认情况下，样本长度至少应为整体上采样因子的整数倍。整体上采样因子等于
        # 2 ** (上采样层数)。不过必要时，可在运行中强制上采样插值的输出尺寸适配任意大小。
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # 当样本不是 `default_overall_up_factor` 的整数倍时，需要向前传递上采样尺寸。
        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # 传递上采样尺寸以强制指定插值输出尺寸。/ Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # 将 attention_mask 转换为偏置，并补上一个单位长度的 query_tokens 维度。
        # 期望输入掩码形状为：
        #   [batch, key_tokens]
        # 补上单位长度的 query_tokens 维度后变为：
        #   [batch,                    1, key_tokens]
        # 这样便于将其作为偏置广播到注意力分数上，注意力分数会是以下形状之一：
        #   [batch,  heads, query_tokens, key_tokens]（例如 torch sdp attn）
        #   [batch * heads, query_tokens, key_tokens]（例如 xformers 或经典 attn）
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # 假设掩码表示为 (1 = 保留, 0 = 丢弃)，将其转换为可加到注意力分数上的偏置：
            # (保留 = +0, 丢弃 = -10000.0)。
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 以与 attention_mask 相同的方式，将 encoder_attention_mask 转换为偏置。
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. 必要时对输入做中心化 / 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. 时间嵌入 / 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        
        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. 预处理 / 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN 位置网络 / 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. 下采样路径 / 3. down
        # 这里用 pop 取出 `scale` 而非 get，否则 `scale` 会传播到内部块并触发弃用警告，
        # 给用户造成困扰。
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # 通过为每个 PEFT 层设置 `lora_scale` 来对 LoRA 层加权。
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # 为 T2I-Adapter 使用新参数 down_intrablock_additional_residuals，以区别于 ControlNet。
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # 兼容旧用法：早期 T2I-Adapter 和 ControlNet 都使用 down_block_additional_residuals 参数，
        # 但二者只能取其一。
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # 适用于 T2I-Adapter 的 CrossAttnDownBlock1D。/ For t2i-adapter CrossAttnDownBlock1D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. 中间块 / 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # 用于支持 T2I-Adapter-XL。/ To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. 上采样路径 / 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # 若尚未到达最后一个块且需要向前传递上采样尺寸，则在此处处理。
            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. 后处理 / 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # 移除每个 PEFT 层上设置的 `lora_scale`。/ remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet1DConditionOutput(sample=sample)