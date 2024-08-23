import functools

import tensorflow as tf

import mlable.ops
import tokun.pipeline

import revml.contract.bytecode

# RESHAPE #####################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

# OFFSET ######################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'00']) + data # HEX 0x00 is a single byte = a single EVM instruction

# MASK ########################################################################

def mask(data: tf.Tensor, padding_value: float=0., padding_weight: float=0., data_weight: float=1., dtype: tf.dtypes.DType=tf.float32) -> tf.Tensor:
    # byte level mask
    __weights = tf.not_equal(data, padding_value)
    # instruction level mask, but expressed byte by byte
    __weights = mlable.ops.reduce_any(data=__weights, group=None, axis=-1, keepdims=False)
    # cast from bool to allow multiplications
    __weights = tf.cast(__weights, dtype=dtype)
    # rescale the weights
    return data_weight * __weights + padding_weight * (1. - __weights)

# TOKENIZE ####################################################################

def _tokenize_data(data: bytes) -> list:
    return (32 - len(data)) * [0] + list(data)[:32]

def _tokenize_opcode(data: bytes) -> list:
    return list(data[:1])

def _tokenize_instruction(data: bytes) -> list:
    return list(data[:1]) + _tokenize_data(data=data[1:])

def _tokenize_bytecode(data: bytes, size: int) -> list:
    __tokenized = [__b for __i in revml.contract.bytecode.iterate_over_instructions(bytecode=data) for __b in _tokenize_instruction(data=__i)]
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
    __length = revml.contract.bytecode.data_length(__opcode)
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

def _preprocess(inputs: tf.Tensor, parser: callable, encoder: callable, embedder: callable, masker: callable, formatter: callable) -> tuple:
    # fetch the relevant features
    __inputs, __contexts, __targets = parser(inputs=inputs)
    # encode / tokenize
    __inputs, __contexts, __targets = encoder(inputs=__inputs, contexts=__contexts, targets=__targets)
    # enforce types + shapes
    __inputs, __contexts, __targets = formatter(inputs=__inputs, contexts=__contexts, targets=__targets)
    # embed with tokun
    __inputs, __contexts, __targets = embedder(inputs=__inputs, contexts=__contexts, targets=__targets)
    # sequence mask to ignore padding during training
    __weights = masker(inputs=__inputs)
    # pack both sourcecode and bytecode into the model inputs
    return ((__inputs, __contexts), __targets, __weights)

def _parser_factory(decoder_config: dict) -> callable:
    def __parser(inputs) -> tuple:
        # fetch the relevant features
        __inputs, __contexts = inputs['creation_bytecode'], inputs['creation_sourcecode']
        # (input, target) where target is the next token for each input
        return (offset(data=__inputs, ticks=decoder_config['token_dim'] // 33), __contexts, __inputs)
    # customized fn
    return __parser

def _encoder_factory(decoder_config: dict, encoder_config: dict) -> callable:
    # bytecode encoding (33 bytes / instruction)
    __encode_i = tokenize_factory(size=decoder_config['sample_dim'], dtype=tf.int32)
    # text encoding (UTF-32-BE)
    __encode_c = functools.partial(tokun.pipeline.encode, token_size=encoder_config['token_dim'], sample_size=encoder_config['sample_dim'], output_dtype=encoder_config['output_dtype'])
    # encode all
    def __encoder(inputs: tf.Tensor, contexts: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__encode_i(inputs), __encode_c(contexts), __encode_i(targets))
    # customized fn
    return __encoder

def _formatter_factory(decoder_config: dict, encoder_config: dict) -> callable:
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.float32)
    __cast_c = functools.partial(tf.cast, dtype=tf.float32)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape_i = functools.partial(tf.reshape, shape=(decoder_config['batch_dim'], (decoder_config['sample_dim'])))
    __reshape_c = functools.partial(tf.reshape, shape=(encoder_config['batch_dim'], (encoder_config['sample_dim'] // 4)))
    __reshape_t = functools.partial(tf.reshape, shape=(decoder_config['batch_dim'], (decoder_config['sample_dim'])))
    # chain the operations
    def __formatter(inputs: tf.Tensor, contexts: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape_i(inputs)), __cast_c(__reshape_c(contexts)), __cast_t(__reshape_t(targets)))
    # customized fn
    return __formatter

def _embedder_factory(decoder_config: dict, encoder_config: dict) -> callable:
    # group bytecode instructions => embeddings
    __embed_i = tokun.model.Tokun(token_dim=decoder_config['token_dim'], input_dim=decoder_config['input_dim'], output_dim=decoder_config['output_dim'], sequence_axis=decoder_config['sequence_axis'], feature_axis=decoder_config['feature_axis'])
    # group sourcecode characters => embeddings
    __embed_c = tokun.model.Tokun(token_dim=encoder_config['token_dim'], input_dim=encoder_config['input_dim'], output_dim=encoder_config['output_dim'], sequence_axis=encoder_config['sequence_axis'], feature_axis=encoder_config['feature_axis'])
    # embed all
    def __embedder(inputs: tf.Tensor, contexts: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__embed_i._encoder(inputs), __embed_c._encoder(contexts), __embed_i._encoder(targets))
    # customized fn
    return __embedder

def _masker_factory(decoder_config: dict) -> callable:
    def __masker(inputs: tf.Tensor) -> tf.Tensor:
        return mask(data=inputs, padding_value=0., data_weight=1., padding_weight=decoder_config['padding_weight'], dtype=tf.float32)
    # customized fn
    return __masker

def preprocess_factory(decoder_config: dict, encoder_config: dict) -> callable:
    # custom fn
    __parser = _parser_factory(decoder_config=decoder_config)
    __encoder = _encoder_factory(decoder_config=decoder_config, encoder_config=encoder_config)
    __embedder = _embedder_factory(decoder_config=decoder_config, encoder_config=encoder_config)
    __formatter = _formatter_factory(decoder_config=decoder_config, encoder_config=encoder_config)
    __masker = _masker_factory(decoder_config=decoder_config)
    # actual preprocessing function
    return functools.partial(_preprocess, parser=__parser, encoder=__encoder, embedder=__embedder, masker=__masker, formatter=__formatter)

# < ###########################################################################

def postprocess(data: tf.Tensor) -> tf.Tensor:
    return data
