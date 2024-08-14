"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.blocks.transformer

# CONSTANTS ###################################################################

EPSILON = 1e-6

# DECODER ONLY ################################################################

@keras.saving.register_keras_serializable(package='blocks')
class SelfDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        sequence_axis: int=1,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(SelfDecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'sequence_axis': sequence_axis,
            'epsilon': epsilon,}
        # layers
        self._attention = mlable.blocks.transformer.SelfAttentionBlock(num_heads=num_heads, head_dim=head_dim, sequence_axis=sequence_axis, epsilon=epsilon)
        self._ffn = mlable.blocks.transformer.FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon)

    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: tf.Tensor=None,
        training: bool=False,
    ) -> tf.Tensor:
        # residual + attention
        __x = inputs + self._attention(inputs=inputs, attention_mask=attention_mask, training=training, use_causal_mask=True)
        # residual + augment
        return __x + self._ffn(__x)

    def get_config(self) -> dict:
        __config = super(SelfDecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
