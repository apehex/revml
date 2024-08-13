import functools

import tensorflow as tf

import mlable.ops
import tokun.pipeline

import revml.contract.decoder.bytecode

# RESHAPE #####################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

# OFFSET ######################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'00']) + data # 0x00 is a single byte = a single EVM instruction

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
    __bytecode = bytes.fromhex(tf.get_static_value(data).decode('utf-8'))
    __data = _tokenize_bytecode(data=__bytecode, size=size)
    return tf.convert_to_tensor(__data, dtype=dtype)

def tokenize_factory(size: int, dtype: tf.dtypes.DType=tf.int32) -> callable:
    # specialized fn
    __fn = functools.partial(_tokenize_scalar, size=size, dtype=dtype)
    # tensorflow wrapper
    @tf.py_function(Tout=dtype)
    def __tokenize(data: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(__fn, data, fn_output_signature=dtype) if int(tf.rank(data)) else __fn(data)
    # return the wrapped function
    return __tokenize

# DETOKENIZE ##################################################################

def _detokenize_instruction(data: list) -> str:
    __opcode = data[0]
    __length = revml.contract.decoder.bytecode.data_length(__opcode)
    __data = data[len(data) - __length:]
    return bytes([__opcode] + __data).hex() if (__opcode > 0) else '' # skip the padding

def _detokenize_bytecode(data: list) -> str:
    __instructions = chunk(seq=data, size=33, repeats= True)
    return ''.join(_detokenize_instruction(__i) for __i in __instructions)

def _detokenize_scalar(data: tf.Tensor) -> tf.Tensor:
    __bytes = tf.get_static_value(data).tolist()
    __data = _detokenize_bytecode(__bytes)
    return tf.convert_to_tensor(__data, dtype=tf.string)

@tf.py_function(Tout=tf.string)
def detokenize(data: tf.Tensor) -> tf.Tensor:
    return _detokenize_scalar(data) if (int(tf.rank(data)) <= 1) else tf.map_fn(_detokenize_scalar, data, fn_output_signature=tf.string)

# > ###########################################################################

def preprocess(inputs: tf.Tensor, token_dim: int, output_dim: int, batch_dim: int, sample_dim: int, padding_weight: float=0., sample_weights: bool=True, binary: bool=True) -> tuple:
    # specialized operations
    __encode_i = tokenize_factory(size=sample_dim, dtype=tf.int32)
    __encode_o = functools.partial(mlable.ops.expand_base, base=2, depth=output_dim) if binary else functools.partial(tf.one_hot, depth=output_dim, axis=-1)
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, sample_dim))
    # (input, target) where target is the next token for each input
    __inputs, __targets = (offset(data=inputs, ticks=token_dim // 33), inputs) # \x00 is one instruction
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
    __weights = mlable.ops.reduce_any(data=__weights, group=33, axis=-1, keepdims=True) # instruction level mask, but expressed byte by byte
    __weights = tf.cast(__weights, dtype=__targets.dtype)
    __weights = __weights + padding_weight * (1. - __weights)
    # chain the operations
    return (__inputs, __targets, __weights) if sample_weights else (__inputs, __targets)

# < ###########################################################################

def postprocess(data: tf.Tensor) -> tf.Tensor:
    return data
