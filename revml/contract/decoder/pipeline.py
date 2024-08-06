import functools

import tensorflow as tf

import mlable.ops
import tokun.pipeline

import revml.contract.decoder.bytecode

# TOKENIZE ####################################################################

def _tokenize_data(data: bytes) -> list:
    return (32 - len(data)) * [0] + list(data)[:32]

def _tokenize_opcode(data: bytes) -> list:
    return list(data[:1])

def _tokenize_instruction(data: bytes) -> list:
    return list(data[:1]) + _tokenize_data(data=data[1:])

def _tokenize_bytecode(data: bytes, size: int) -> list:
    __tokenized = [__b for __i in revml.contract.decoder.bytecode.iterate_over_instructions(bytecode=data) for __b in _tokenize_instruction(data=__i)]
    return __tokenized[:size] + (size - len(__tokenized)) * [0]

def _tokenize_scalar(data: tf.Tensor, size: int, dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    __data = _tokenize_bytecode(data=bytes.fromhex(data.numpy()), size=size)
    return tf.convert_to_tensor(__data, dtype=dtype)

def tokenize(data: tf.Tensor, size: int, dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    __fn = functools.partial(_tokenize_scalar, size=size, dtype=dtype)
    return tf.map_fn(__fn, data, fn_output_signature=dtype)

# DETOKENIZE ##################################################################

def _detokenize_instruction(data: list) -> list:
    return data

def _detokenize_scalar(data: tf.Tensor) -> list:
    return data

def detokenize(data: tf.Tensor) -> tf.Tensor:
    return data

# > ###########################################################################

def preprocess(inputs: tf.Tensor, token_dim: int, output_dim: int, batch_dim: int, sample_dim: int, padding_weight: float=0., sample_weights: bool=True, binary: bool=True) -> tuple:
    # specialized operations
    __encode_i = functools.partial(tokenize, size=sample_dim, dtype=tf.int32)
    __encode_o = functools.partial(mlable.ops.expand_base, base=2, depth=output_dim) if binary else functools.partial(tf.one_hot, depth=output_dim, axis=-1)
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, sample_dim))
    # (input, target) where target is the next token for each input
    __inputs, __targets = (tokun.pipeline.offset(data=inputs, ticks=token_dim // 33), inputs) # \x00 is one instruction
    # tokenize => (B, 33 * T) = (B, S) int
    __inputs, __targets = (__encode_i(__inputs), __encode_i(__targets))
    # enforce shapes
    __inputs, __targets = (__reshape(__inputs), __reshape(__targets))
    # binary / categorical encoding for the target classes
    __inputs, __targets = __inputs, __encode_o(__targets)
    # enforce types
    __inputs, __targets = tf.cast(__inputs, dtype=tf.dtypes.int32), tf.cast(__targets, dtype=tf.dtypes.float32)
    # sequence mask to ignore padding during training
    __weights = tf.not_equal(__inputs, 0) # byte level mask
    __weights = mlable.ops.reduce_any(data=__weights, group=token_dim, axis=-1, keepdims=True) # instruction level mask, but expressed byte by byte
    __weights = tf.cast(__weights, dtype=__targets.dtype)
    __weights = __weights + padding_weight * (1. - __weights)
    # chain the operations
    return (__inputs, __targets, __weights) if sample_weights else (__inputs, __targets)

# < ###########################################################################

def postprocess(data: tf.Tensor) -> tf.Tensor:
    return data
