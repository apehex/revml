"""llaminate model."""

import functools

import keras
import tensorflow as tf

import mlable.layers.embedding

import revml.contract.layers

# CONSTANTS ###################################################################

EPSILON = 1e-6

# ENCODING-DECODING ##############################################################

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        input_dim: int,
        context_dim: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Transformer, self).__init__(**kwargs)
        # config
        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'input_dim': input_dim,
            'context_dim': context_dim,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'epsilon': epsilon,}
        # the inputs is always UTF-32-BE bytes => 256
        self._embed_input = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=embed_dim // input_dim, name='embed-inputs')
        self._embed_context = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=embed_dim // context_dim, name='embed-contexts')
        # blocks
        self._blocks = [
            revml.contract.layers.DecoderBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                sequence_axis=1,
                epsilon=epsilon,
                name='block-{}'.format(__i))
            for __i in range(num_layers)]
        # 8 bits for each input byte
        self._project = tf.keras.layers.Dense(units=8 * input_dim, activation='sigmoid', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='project')
        # group by token
        self._divide = mlable.layers.reshaping.Divide(input_axis=-2, output_axis=-1, factor=8, insert=True, name='divide')

    def call(self, inputs: tuple, attention_mask: tf.Tensor=None, **kwargs) -> tf.Tensor:
        # unpack
        __inputs, __contexts = inputs
        # embed
        __y = self._embed_input(__inputs)
        __c = self._embed_context(__contexts)
        # blocks
        __y = functools.reduce(lambda __x, __b: __b(inputs=__x, contexts=__c, attention_mask=attention_mask, **kwargs), self._blocks, __y)
        # decompress
        return self._divide(self._project(__y))

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
