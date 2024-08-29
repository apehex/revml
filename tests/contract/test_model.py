import functools
import math

import tensorflow as tf

import mlable.sampling
import tokun.model

import revml.contract.pipeline
import revml.contract.model

# RAW PREDICTIONS #############################################################

class RawTransformerTest(tf.test.TestCase):
    def setUp(self):
        super(RawTransformerTest, self).setUp()
        # preprocessing config
        self._config_encoder = {'batch_dim': 4, 'sample_dim': 64 * 4 * 128, 'token_dim': 64, 'input_dim': 0x40000, 'output_dim': 0x40000, 'sequence_axis': 1, 'feature_axis': -1, 'output_dtype': tf.int32,}
        self._config_decoder = {'batch_dim': 4, 'sample_dim': 33 * 4 * 128, 'token_dim': 33 * 4, 'input_dim': 256, 'output_dim': 256, 'sequence_axis': 1, 'feature_axis': -1, 'data_weight': 1.0, 'padding_weight': 0., 'binary': False,}
        self._config_model = {'num_layers': 4, 'num_heads': 4, 'embed_dim': 64, 'head_dim': 16, 'hidden_dim': 256,  'output_dim': 33 * 4,}
        # specialized preprocessing fn
        self._preprocess = revml.contract.pipeline.preprocess_factory(decoder_config=self._config_decoder, encoder_config=self._config_encoder)
        # original dataset
        __b0 = b'3d602d80600a3d3981f3363d3d373d3d3d363d7339778bc77bd7a9456655b19fd4c5d0bf2071104e5af43d82803e903d91602b57fd5bf3'
        __b1 = b'7f602036038060203d373d3d3d923d343d355af13d82803e903d91601e57fd5bf33d5260203df3'
        __b2 = b'60a060405234801561001057600080fd5b506040516101093803806101098339818101604052602081101561003357600080fd5b50516001600160601b031960609190911b16608052600080546001600160a01b0319163317905560805160601c609561007460003980601b525060956000f3fe608060405236600a57005b348015601557600080fd5b506040517f0000000000000000000000000000000000000000000000000000000000000000903680600083376000808284865af43d9150816000843e808015605b578284f35b8284fdfea2646970667358221220c4a46717f67197616503d87ff2612122c70a6a9ed6d1f1999e000b68e428470964736f6c634300060c003300000000000000000000000080d0f44d6c1563de6ba356aa8dfe7abdbe8a174a'
        __b3 = b'608060405234801561001057600080fd5b506040516101e63803806101e68339818101604052602081101561003357600080fd5b8101908080519060200190929190505050600073ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff1614156100ca576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004018080602001828103825260228152602001806101c46022913960400191505060405180910390fd5b806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505060ab806101196000396000f3fe608060405273ffffffffffffffffffffffffffffffffffffffff600054167fa619486e0000000000000000000000000000000000000000000000000000000060003514156050578060005260206000f35b3660008037600080366000845af43d6000803e60008114156070573d6000fd5b3d6000f3fea2646970667358221220d1429297349653a4918076d650332de1a1068c5f3e07c5c82360c277770b955264736f6c63430007060033496e76616c69642073696e676c65746f6e20616464726573732070726f7669646564000000000000000000000000d9db270c1b5e3bd161e8c8503c55ceabee709552'
        __b4 = b'900e60ffbb1010136ef09f3bf7356866882f5b9b56e9fbdbb373d44480469538'
        __b5 = b'2199ef7a217e1a55c1dfff22eb3a60af6fcdc91126d1dc7cbd6711c50a624cc0'
        __b6 = b'ce83d7b0b2a7fd96dc7ca8bf02d85b6b9acd71641f20080326d37421c296e9e2'
        __b7 = b'60008060008034415af1'
        self._dataset_before = tf.data.Dataset.from_tensor_slices({
            'creation_bytecode': [__b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7,],
            'creation_sourcecode': 8 * [1024 * b'a']})
        # preprocessed datasets
        self._dataset_after = self._dataset_before.batch(self._config_decoder['batch_dim'], drop_remainder=True).map(self._preprocess)
        # transformer
        self._model = revml.contract.model.Transformer(**self._config_model)
        # build
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_decoder['token_dim']], dtype=tf.float32)
        __c = tf.zeros([self._config_encoder['batch_dim'], (self._config_encoder['sample_dim'] // (4 * self._config_encoder['token_dim'])), self._config_encoder['token_dim']], dtype=tf.float32)
        self._model((__x, __c))

    def test_internals(self):
        # tail
        assert list(self._model._tail.kernel.shape) == [self._config_decoder['token_dim'], self._config_model['embed_dim']]
        assert list(self._model._tail.bias.shape) == [self._config_model['embed_dim']]
        # blocks
        assert len(self._model._blocks) == self._config_model['num_layers']
        # self attention
        assert all(bool(__b._self_attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._self_attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._self_attention._position) for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._key_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._value_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # cross attention
        assert all(bool(__b._cross_attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._cross_attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._cross_attention._position) for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._key_dense.kernel.shape) == [self._config_encoder['token_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._value_dense.kernel.shape) == [self._config_encoder['token_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # ffn
        assert all(bool(__b._ffn._norm) for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._gelu.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._linear.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._output.kernel.shape) == [self._config_model['hidden_dim'], self._config_model['embed_dim']] for __b in self._model._blocks)
        # head
        assert list(self._model._head.kernel.shape) == [self._config_model['embed_dim'], self._config_model['output_dim']]
        # assert list(self._model._head.bias.shape) == [self._config_model['output_dim']]

    def test_shapes(self):
        __batch = iter(self._dataset_after)
        for _ in range(2):
            (__x, __c), __t, __w = next(__batch)
            assert list(self._model((__x, __c)).shape) == [self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['output_dim']]

    def test_null_values(self):
        # tail
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_decoder['token_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._tail(__x), 0.5 * tf.ones([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # self attention
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._self_attention(__x), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # cross attention
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        __c = tf.zeros([self._config_encoder['batch_dim'], (self._config_encoder['sample_dim'] // (4 * self._config_encoder['token_dim'])), self._config_encoder['token_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._cross_attention(inputs=__x, contexts=__c), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # ffn
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._ffn(__x), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # head
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._head(__x), 0.5 * tf.ones([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['output_dim']], dtype=tf.float32))

# RAW PREDICTIONS #############################################################

class BinaryTransformerTest(tf.test.TestCase):
    def setUp(self):
        super(BinaryTransformerTest, self).setUp()
        # preprocessing config
        self._config_encoder = {'batch_dim': 4, 'sample_dim': 64 * 4 * 128, 'token_dim': 64 * 4, 'input_dim': 256, 'output_dim': 256, 'sequence_axis': 1, 'feature_axis': -1, 'output_dtype': tf.uint8,}
        self._config_decoder = {'batch_dim': 4, 'sample_dim': 33 * 4 * 128, 'token_dim': 33 * 4, 'input_dim': 256, 'output_dim': 256, 'sequence_axis': 1, 'feature_axis': -1, 'data_weight': 1.0, 'padding_weight': 0., 'binary': True,}
        self._config_model = {'num_layers': 4, 'num_heads': 4, 'embed_dim': 256, 'head_dim': 64, 'hidden_dim': 1024,  'output_dim': 33 * 4 * 8,}
        # length of each encoded value in bytes
        self._factor_encoder = 1 if self._config_encoder['output_dtype'] == tf.uint8 else 4
        # specialized preprocessing fn
        self._preprocess = revml.contract.pipeline.preprocess_factory(decoder_config=self._config_decoder, encoder_config=self._config_encoder)
        # original dataset
        __b0 = b'3d602d80600a3d3981f3363d3d373d3d3d363d7339778bc77bd7a9456655b19fd4c5d0bf2071104e5af43d82803e903d91602b57fd5bf3'
        __b1 = b'7f602036038060203d373d3d3d923d343d355af13d82803e903d91601e57fd5bf33d5260203df3'
        __b2 = b'60a060405234801561001057600080fd5b506040516101093803806101098339818101604052602081101561003357600080fd5b50516001600160601b031960609190911b16608052600080546001600160a01b0319163317905560805160601c609561007460003980601b525060956000f3fe608060405236600a57005b348015601557600080fd5b506040517f0000000000000000000000000000000000000000000000000000000000000000903680600083376000808284865af43d9150816000843e808015605b578284f35b8284fdfea2646970667358221220c4a46717f67197616503d87ff2612122c70a6a9ed6d1f1999e000b68e428470964736f6c634300060c003300000000000000000000000080d0f44d6c1563de6ba356aa8dfe7abdbe8a174a'
        __b3 = b'608060405234801561001057600080fd5b506040516101e63803806101e68339818101604052602081101561003357600080fd5b8101908080519060200190929190505050600073ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff1614156100ca576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004018080602001828103825260228152602001806101c46022913960400191505060405180910390fd5b806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505060ab806101196000396000f3fe608060405273ffffffffffffffffffffffffffffffffffffffff600054167fa619486e0000000000000000000000000000000000000000000000000000000060003514156050578060005260206000f35b3660008037600080366000845af43d6000803e60008114156070573d6000fd5b3d6000f3fea2646970667358221220d1429297349653a4918076d650332de1a1068c5f3e07c5c82360c277770b955264736f6c63430007060033496e76616c69642073696e676c65746f6e20616464726573732070726f7669646564000000000000000000000000d9db270c1b5e3bd161e8c8503c55ceabee709552'
        __b4 = b'900e60ffbb1010136ef09f3bf7356866882f5b9b56e9fbdbb373d44480469538'
        __b5 = b'2199ef7a217e1a55c1dfff22eb3a60af6fcdc91126d1dc7cbd6711c50a624cc0'
        __b6 = b'ce83d7b0b2a7fd96dc7ca8bf02d85b6b9acd71641f20080326d37421c296e9e2'
        __b7 = b'60008060008034415af1'
        self._dataset_before = tf.data.Dataset.from_tensor_slices({
            'creation_bytecode': [__b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7,],
            'creation_sourcecode': 8 * [1024 * b'a']})
        # preprocessed datasets
        self._dataset_after = self._dataset_before.batch(self._config_decoder['batch_dim'], drop_remainder=True).map(self._preprocess)
        # transformer
        self._model = revml.contract.model.Transformer(**self._config_model)
        # build
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_decoder['token_dim']], dtype=tf.float32)
        __c = tf.zeros([self._config_encoder['batch_dim'], (self._config_encoder['sample_dim'] // (self._factor_encoder * self._config_encoder['token_dim'])), self._config_encoder['token_dim']], dtype=tf.float32)
        self._model((__x, __c))

    def test_internals(self):
        # tail
        assert list(self._model._tail.kernel.shape) == [self._config_decoder['token_dim'], self._config_model['embed_dim']]
        assert list(self._model._tail.bias.shape) == [self._config_model['embed_dim']]
        # blocks
        assert len(self._model._blocks) == self._config_model['num_layers']
        # self attention
        assert all(bool(__b._self_attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._self_attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._self_attention._position) for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._key_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._self_attention._attention._value_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # cross attention
        assert all(bool(__b._cross_attention._input_norm) for __b in self._model._blocks)
        assert all(bool(__b._cross_attention._context_norm) for __b in self._model._blocks)
        assert all(bool(__b._cross_attention._position) for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._key_dense.kernel.shape) == [self._config_encoder['token_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._query_dense.kernel.shape) == [self._config_model['embed_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        assert all(list(__b._cross_attention._attention._value_dense.kernel.shape) == [self._config_encoder['token_dim'], self._config_model['num_heads'], self._config_model['head_dim']] for __b in self._model._blocks)
        # ffn
        assert all(bool(__b._ffn._norm) for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._gelu.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._linear.kernel.shape) == [self._config_model['embed_dim'], self._config_model['hidden_dim']] for __b in self._model._blocks)
        assert all(list(__b._ffn._ffn._output.kernel.shape) == [self._config_model['hidden_dim'], self._config_model['embed_dim']] for __b in self._model._blocks)
        # head
        assert list(self._model._head.kernel.shape) == [self._config_model['embed_dim'], self._config_model['output_dim']]
        # assert list(self._model._head.bias.shape) == [self._config_model['output_dim']]

    def test_shapes(self):
        __batch = iter(self._dataset_after)
        for _ in range(2):
            (__x, __c), __t, __w = next(__batch)
            assert list(self._model((__x, __c)).shape) == [self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['output_dim']]
            assert list(self._model((__x, __c)).shape) == list(__t.shape)

    def test_null_values(self):
        # tail
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_decoder['token_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._tail(__x), 0.5 * tf.ones([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # self attention
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._self_attention(__x), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # cross attention
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        __c = tf.zeros([self._config_encoder['batch_dim'], (self._config_encoder['sample_dim'] // (4 * self._config_encoder['token_dim'])), self._config_encoder['token_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._cross_attention(inputs=__x, contexts=__c), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # ffn
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._blocks[0]._ffn(__x), tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32))
        # head
        __x = tf.zeros([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['embed_dim']], dtype=tf.float32)
        self.assertAllEqual(self._model._head(__x), 0.5 * tf.ones([self._config_decoder['batch_dim'], (self._config_decoder['sample_dim'] // self._config_decoder['token_dim']), self._config_model['output_dim']], dtype=tf.float32))
