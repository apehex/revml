import functools

import tensorflow as tf

import contract.decoder.bytecode

# TOKENIZE ####################################################################

def _tokenize_data(data: bytes) -> list:
    return (32 - len(data)) * [0] + list(data)[:32]

def _tokenize_opcode(data: bytes) -> list:
    return list(data[:1])

def _tokenize_instruction(data: bytes) -> list:
    return list(data[:1]) + _tokenize_data(data=data[1:])

def _tokenize_bytecode(data: bytes, size: int) -> list:
    __tokenized = [__b for __i in contract.decoder.bytecode.iterate_over_instructions(bytecode=data) for __b in _tokenize_instruction(data=__i)]
    return __tokenized[:size] + (size - len(__tokenized)) * [0]

def _tokenize_scalar(data: tf.Tensor, size: int, dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    __data = _tokenize_bytecode(data=data.numpy(), size=size)
    return tf.convert_to_tensor(__data, dtype=dtype)

def tokenize(data: tf.Tensor, size: int, dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    __fn = functools.partial(_tokenize_scalar, size=size, dtype=dtype)
    return tf.map_fn(__fn, data, fn_output_signature=dtype)

# DETOKENIZE ##################################################################

def _detokenize_instruction(data: list) -> list:
    return data

def _detokenize_scalar(data: tf.tensor) -> list:
    return data

def detokenize(data: tf.Tensor) -> tf.Tensor:
    return data

# > ###########################################################################

def preprocess(data: tf.Tensor) -> tf.Tensor:
    return data

# < ###########################################################################

def postprocess(data: tf.Tensor) -> tf.Tensor:
    return data
