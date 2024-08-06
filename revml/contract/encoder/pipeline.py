import functools

import tensorflow as tf

import mlable.ops
import tokun.pipeline

# PREPROCESS ##################################################################

def preprocess(inputs: tf.Tensor, token_dim: int, output_dim: int, batch_dim: int, sample_dim: int, padding_weight: float=0., sample_weights: bool=True, binary: bool=True) -> tuple:
    # specialized operations
    __encode_i = functools.partial(tokun.pipeline.encode, token_size=token_dim, sample_size=sample_dim)
    __encode_o = functools.partial(mlable.ops.expand_base, base=2, depth=output_dim) if binary else functools.partial(tf.one_hot, depth=output_dim, axis=-1)
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, 4 * sample_dim))
    # encode => (B, 4 * S,) int
    __inputs = __encode_i(inputs)
    # reshape => (B, 4 * S,) int
    __inputs = __reshape(__inputs)
    # binary encoding for the target classes
    __inputs, __targets = (__inputs, __encode_o(__inputs))
    # enforce types
    __inputs, __targets = tf.cast(__inputs, dtype=tf.dtypes.int32), tf.cast(__targets, dtype=tf.dtypes.float32)
    # sequence mask to ignore padding during training
    __weights = tf.not_equal(__inputs, 0) # byte level mask
    __weights = mlable.ops.reduce_any(data=__weights, group=4, axis=-1, keepdims=True) # character level mask, but expressed byte by byte
    __weights = tf.cast(__weights, dtype=__targets.dtype)
    __weights = __weights + padding_weight * (1. - __weights)
    # chain the operations
    return (__inputs, __targets, __weights) if sample_weights else (__inputs, __targets)
