# 改编自 HuggingFace diffusers / Adapted from HuggingFace diffusers
# https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/transformers/transformer_2d.py

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import LegacyConfigMixin, register_to_config
from diffusers.utils import deprecate, is_torch_version, logging
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection
from ..modeling_outputs import Transformer1DModelOutput
from diffusers.models.modeling_utils import LegacyModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Transformer1DModel(LegacyModelMixin, LegacyConfigMixin):
    """
    用于序列数据的 1D Transformer 模型 / A 1D Transformer model for seq data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): 多头注意力的头数 / The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): 每个注意力头的通道数 / The number of channels in each head.
        in_channels (`int`, *optional*):
            输入与输出的通道数（当输入为 **连续型** 时指定） /
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): 使用的 Transformer block 层数 / The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): 使用的 dropout 概率 / The dropout probability to use.
        cross_attention_dim (`int`, *optional*): `encoder_hidden_states` 的维度数 / The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): 隐空间图像的宽度（当输入为 **离散型** 时指定） / The width of the latent images (specify if the input is **discrete**).
            训练期间固定，因为它用于学习一定数量的位置嵌入。 /
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            隐空间像素向量嵌入的类别数（当输入为 **离散型** 时指定），包含被遮挡隐空间像素的类别。 /
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): 前馈网络中使用的激活函数 / Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            训练期间使用的扩散步数。当至少有一个 norm 层为 `AdaLayerNorm` 时需要传入。训练期间固定，
            因为它用于学习一定数量、并加到 hidden states 上的嵌入。 /
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            推理时去噪步数最多可达 `num_embeds_ada_norm`，但不能超过该值。 /
            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            配置 `TransformerBlocks` 的注意力是否包含 bias 参数。 /
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 可选/options: 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
        use_additional_conditions: Optional[bool] = None,
    ):
        super().__init__()

        # 1. Transformer1DModel 仅支持处理连续型输入
        # 1. Transformer1DModel can only process continuous input
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        assert self.is_input_continuous
        self.is_input_vectorized = False
        self.is_input_patches = False

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 设置一些贯穿全局使用的通用变量
        # Set some common variables used across the board.
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale = interpolation_scale
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        if use_additional_conditions is None:
            if norm_type == "ada_norm_single" and sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 2. 初始化对应的网络模块。
        #    这些函数遵循统一的结构：
        #    a. 初始化输入模块。 b. 初始化 transformer 模块。
        #    c. 必要时初始化输出模块及其他投影模块。
        # 2. Initialize the right blocks.
        #    These functions follow a common structure:
        #    a. Initialize the input blocks. b. Initialize the transformer blocks.
        #    c. Initialize the output blocks and other projection blocks when necessary.
        if self.is_input_continuous:
            self._init_continuous_input(norm_type=norm_type)

    def _init_continuous_input(self, norm_type):
        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            self.proj_in = torch.nn.Conv1d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = torch.nn.Conv1d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        [`Transformer2DModel`] 的前向传播方法 / The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, length)`):
                输入的 `hidden_states`。 / Input `hidden_states`.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                交叉注意力层的条件嵌入。若未提供，交叉注意力会退化为自注意力。 /
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                用于指示去噪时间步。可选的时间步，将作为嵌入应用于 `AdaLayerNorm`。 /
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                用于指示类别标签条件。可选的类别标签，将作为嵌入应用于 `AdaLayerZeroNorm`。 /
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                一个 kwargs 字典；若指定，会传递给在如下位置 `self.processor` 中定义的 `AttentionProcessor`： /
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                形状为 `(batch, key_tokens)` 的注意力掩码，应用于 `encoder_hidden_states`。值为 `1` 保留掩码，
                值为 `0` 则丢弃。掩码会被转换为 bias，向对应 "丢弃" token 的注意力分数加上较大的负值。 /
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                应用于 `encoder_hidden_states` 的交叉注意力掩码。支持两种格式： /
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * 掩码 `(batch, sequence_length)` True = 保留，False = 丢弃。 /
                      Mask `(batch, sequence_length)` True = keep, False = discard.
                    * 偏置 `(batch, 1, sequence_length)` 0 = 保留，-10000 = 丢弃。 /
                      Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                若 `ndim == 2`：会被解释为掩码，再按上述格式转换为 bias，该 bias 将被加到交叉注意力分数上。 /
                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回 [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] 而非普通 tuple。 /
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            若 `return_dict` 为 True，返回 [`~models.transformers.transformer_2d.Transformer2DModelOutput`]；
            否则返回一个 `tuple`，其第一个元素为 sample 张量。 /
            If `return_dict` is True, an [`~models.transformers.transformer_2d.Transformer2DModelOutput`] is returned,
            otherwise a `tuple` where the first element is the sample tensor.
        """
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # 确保 attention_mask 是 bias 形式，并为其增加一个单元的 query_tokens 维度。
        #   该转换可能已经完成，例如经由 UNet2DConditionModel#forward 进入此处。
        #   可通过维度数判断：若 ndim == 2，则它是掩码而非 bias。
        # 期望的掩码形状：
        #   [batch, key_tokens]
        # 增加单元的 query_tokens 维度后：
        #   [batch,                    1, key_tokens]
        # 这样便于将其作为 bias 广播到注意力分数上，注意力分数会是以下形状之一：
        #   [batch,  heads, query_tokens, key_tokens]（例如 torch sdp attn）
        #   [batch * heads, query_tokens, key_tokens]（例如 xformers 或经典 attn）
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # 假设掩码表示为：(1 = 保留，0 = 丢弃)
            # 将掩码转换为可加到注意力分数上的 bias：(保留 = +0，丢弃 = -10000.0)
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 以处理 attention_mask 同样的方式，将 encoder_attention_mask 转换为 bias
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. 输入 / Input
        if self.is_input_continuous:
            batch_size, _, length = hidden_states.shape
            residual = hidden_states
            # hidden_states 形状变换：[B, C, L] -> [B, L, C]
            # hidden_states shape: [B, C, L] -> [B, L, C]
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)

        # 2. Transformer 模块 / Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. 输出 / Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                length=length,
                inner_dim=inner_dim,
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, length = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 1)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim

    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, length, inner_dim):
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch_size, length, inner_dim).permute(0, 2, 1).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_size, length, inner_dim).permute(0, 2, 1).contiguous()
            )

        output = hidden_states + residual
        return output