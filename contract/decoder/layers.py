"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.transformer

# CONSTANTS ###################################################################

EPSILON = 1e-5

# WITHOUT CACHE ###############################################################

@keras.saving.register_keras_serializable(package='blocks')
class DecoderBlock(tf.keras.layers.Layer):
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
        super(DecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # layers
        self._attention_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, beta_initializer='zeros', gamma_initializer='ones') # rms_scaling=True
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=1, feature_axis=-1)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=False, kernel_initializer='glorot_uniform')
        self._ffn_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, beta_initializer='zeros', gamma_initializer='ones') # rms_scaling=True
        self._ffn = mlable.layers.transformer.FeedForwardGate(input_dim=embed_dim, hidden_dim=hidden_dim)

    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: tf.Tensor=None,
        training: bool=False,
    ) -> tf.Tensor:
        # residual
        __x = inputs
        # normalize
        __y = self._attention_norm(__x)
        # position embedding
        __yp = self._position(inputs=__y, offset=0)
        # attention
        __y = self._attention(key=__yp, query=__yp, value=__y, training=training, attention_mask=attention_mask, use_causal_mask=True, return_attention_scores=False)
        # residual
        __x = __y + __x
        # normalize
        __y = self._ffn_norm(__x)
        # augment
        __y = self._ffn(__y)
        # residual
        return __y + __x

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
