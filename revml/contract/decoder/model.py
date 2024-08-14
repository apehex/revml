"""llaminate model."""

import functools

import keras
import tensorflow as tf

import tokun.model

import revml.contract.decoder.layers

# CONSTANTS ###################################################################

EPSILON = 1e-6

# DECODING ONLY ###############################################################

@keras.saving.register_keras_serializable(package='models')
class SelfTransformer(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        input_dim: int=256,
        output_dim: int=8,
        token_dim: list=[33],
        epsilon: float=EPSILON,
        activation: str='gelu',
        output: str='binary',
        **kwargs
    ) -> None:
        # init
        super(SelfTransformer, self).__init__(**kwargs)
        # config
        self._config = {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'embed_dim': embed_dim,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'token_dim': token_dim,
            'epsilon': epsilon,
            'activation': activation,
            'output': output,}
        # layers
        self._encoder = tokun.model.Encoder(token_dim=token_dim, encoding_dim=input_dim, embedding_dim=embed_dim, sequence_axis=1, feature_axis=-1, activation=activation, name='encoder')
        self._blocks = [
            revml.contract.decoder.layers.SelfDecoderBlock(
                num_heads=num_heads,
                embed_dim=embed_dim,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                sequence_axis=1,
                epsilon=epsilon,
                name='block-{}'.format(__i))
            for __i in range(num_layers)]
        self._decoder = tokun.model.Decoder(token_dim=token_dim[::-1], encoding_dim=output_dim, embedding_dim=embed_dim, sequence_axis=1, feature_axis=-1, activation=activation, output=output, name='decoder')

    def call(self, inputs: tf.Tensor, attention_mask: tf.Tensor=None, **kwargs) -> tf.Tensor:
        # compress the inputs
        __y = self._encoder(inputs)
        # blocks
        __y = functools.reduce(lambda __x, __b: __b(inputs=__x, attention_mask=attention_mask, **kwargs), self._blocks, __y)
        # decompress
        return self._decoder(__y)

    def get_config(self) -> dict:
        __config = super(SelfTransformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
