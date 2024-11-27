"""Building blocks of llaminate."""

import keras
import tensorflow as tf

import mlable.blocks.transformer

# CONSTANTS ####################################################################

EPSILON = 1e-6
DROPOUT = 0.0

# TRANSFORMER ##################################################################

@keras.saving.register_keras_serializable(package='blocks')
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        head_num: int,
        attention_axes: list=[1],
        expand_rate: int=4,
        dropout_rate: float=DROPOUT,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(TransformerBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'head_num': head_num,
            'attention_axes': [attention_axes] if isinstance(attention_axes, int) else list(attention_axes),
            'expand_rate': int(expand_rate),
            'dropout_rate': dropout_rate,
            'epsilon': epsilon,}
        # layers
        self._attend_cross = None
        self._attend_self = None
        # custom build
        self._built = False

    def _build(self, inputs_shape: tuple, contexts_shape: tuple) -> None:
        if not self._built:
            __embed_dim = inputs_shape[-1]
            __head_num = self._config['head_num']
            __head_dim = __embed_dim // __head_num
            __hidden_dim = __embed_dim * self._config['expand_rate']
            # common args
            __args = {
                'head_num': __head_num,
                'key_dim': __head_dim,
                'value_dim': __head_dim,
                'hidden_dim': __hidden_dim,
                'attention_axes': self._config['attention_axes'],
                'dropout_rate': self._config['dropout_rate'],
                'epsilon': self._config['epsilon'],
                'use_bias': True,
                'center': True,
                'scale': True,}
            # init
            self._attend_cross = mlable.blocks.transformer.ResidualDecoderBlock(**__args)
            self._attend_self = mlable.blocks.transformer.ResidualDecoderBlock(**__args)
            # propagate the shapes trhough the child layers
            self._attend_cross._build(query_shape=inputs_shape, key_shape=contexts_shape, value_shape=contexts_shape)
            self._attend_self._build(query_shape=inputs_shape, key_shape=inputs_shape, value_shape=inputs_shape)
            # register
            self.built, self._built = True, True

    def build(self, inputs_shape: tuple, contexts_shape: tuple=None) -> None:
        if contexts_shape is not None:
            self._build(inputs_shape=inputs_shape, contexts_shape=contexts_shape)

    def call(
        self,
        inputs: tf.Tensor,
        contexts: tf.Tensor,
        attention_mask: tf.Tensor=None,
        training: bool=False,
        **kwargs
    ) -> tf.Tensor:
        self._build(inputs_shape=tuple(inputs.shape), contexts_shape=tuple(contexts.shape))
        # residual + cross attention + FFN
        __outputs = self._attend_cross(query=inputs, key=contexts, value=contexts, attention_mask=None, training=training, use_causal_mask=False)
        # residual + self attention + FFN
        return self._attend_self(query=__outputs, key=__outputs, value=__outputs, attention_mask=attention_mask, training=training, use_causal_mask=True)

    def get_config(self) -> dict:
        __config = super(TransformerBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
