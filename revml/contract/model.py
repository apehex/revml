"""llaminate model."""

import functools

import keras
import tensorflow as tf

import mlable.layers.embedding

import revml.contract.layers

# CONSTANTS ###################################################################

EPSILON = 1e-6
DROPOUT = 0.0

# ENCODING-DECODING ##############################################################

@keras.saving.register_keras_serializable(package='models')
class Transformer(tf.keras.models.Model):
    def __init__(
        self,
        layer_num: int,
        head_num: int,
        input_dim: int,
        context_dim: int,
        embed_dim: int,
        expand_rate: int=4,
        dropout_rate: float=DROPOUT,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Transformer, self).__init__(**kwargs)
        # config
        self._config = {
            'layer_num': layer_num,
            'head_num': head_num,
            'input_dim': input_dim,
            'context_dim': context_dim,
            'embed_dim': embed_dim,
            'expand_rate': expand_rate,
            'dropout_rate': dropout_rate,
            'epsilon': epsilon,}
        # layers
        self._encoder_input = None
        self._encoder_context = None
        self._transformer = []
        self._decoder = None

    def build(self, input_shape: tuple) -> None:
        __inputs_shape, __contexts_shape = list(input_shape[0]), list(input_shape[-1])
        # the inputs is always UTF-32-BE bytes => 256
        self._encoder_input = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=self._config['embed_dim'] // self._config['input_dim'], name='encoder-input')
        self._encoder_context = mlable.layers.embedding.TokunEmbedding(input_dim=256, output_dim=self._config['embed_dim'] // self._config['context_dim'], name='encoder-context')
        # blocks
        self._transformer = [
            revml.contract.layers.TransformerBlock(
                attention_axes=[1],
                head_num=self._config['head_num'],
                expand_rate=self._config['expand_rate'],
                dropout_rate=self._config['dropout_rate'],
                epsilon=self._config['epsilon'],
                name='block-{}'.format(__i))
            for __i in range(self._config['layer_num'])]
        # 8 bits for each input byte
        self._decoder = tf.keras.layers.Dense(units=8 * self._config['input_dim'], activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decoder')
        # the embeddings are entirely defined in the constructor
        self._encoder_input.build(__inputs_shape)
        self._encoder_context.build(__contexts_shape)
        # both inputs and contexts have the same feature dimension after embedding
        __inputs_shape[-1] = self._config['embed_dim']
        __contexts_shape[-1] = self._config['embed_dim']
        # propagate the shapes through the child layers
        for __b in self._transformer: __b.build(inputs_shape=__inputs_shape, contexts_shape=__contexts_shape)
        self._decoder.build(__inputs_shape)
        # register
        self.built = True

    def call(self, inputs: tuple, attention_mask: tf.Tensor=None, training: bool=False, **kwargs) -> tf.Tensor:
        # unpack
        __inputs, __contexts = inputs
        # embed
        __y = self._encoder_input(__inputs)
        __c = self._encoder_context(__contexts)
        # blocks
        __y = functools.reduce(lambda __x, __b: __b(inputs=__x, contexts=__c, attention_mask=attention_mask, training=training, **kwargs), self._transformer, __y)
        # decompress
        return self._decoder(__y)

    def get_config(self) -> dict:
        __config = super(Transformer, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
