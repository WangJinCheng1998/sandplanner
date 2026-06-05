from dataclasses import dataclass

from diffusers.utils import BaseOutput

@dataclass
class Transformer1DModelOutput(BaseOutput):
    """
    [`Transformer1DModel`] 的输出。 / The output of [`Transformer1DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, length)`):
            基于 `encoder_hidden_states` 输入条件化得到的隐藏状态输出。 /
            The hidden states output conditioned on the `encoder_hidden_states` input.
    """

    sample: "torch.Tensor"