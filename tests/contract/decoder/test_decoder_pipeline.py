import functools

import tensorflow as tf

import mlable.sampling
import revml.contract.decoder.pipeline

# TOKENIZATION ################################################################

class TokenizeTest(tf.test.TestCase):
    def setUp(self):
        super(TokenizeTest, self).setUp()
        # scalar inputs
        self._b0 = bytes.fromhex('6eb3f879cb30fe243b4dfee438691c043318585733ff')
        self._b1 = bytes.fromhex('756eb3f879cb30fe243b4dfee438691c043318585733ff6000526016600af3')
        self._x0 = tf.convert_to_tensor(self._b0.hex(), dtype=tf.string)
        self._x1 = tf.convert_to_tensor(self._b1.hex(), dtype=tf.string)
        # fn
        self._fn = revml.contract.decoder.pipeline.tokenize_factory(size=33 * 8, dtype=tf.uint8)
        # tokenized
        self._t0 = self._fn(self._x0)
        self._t1 = self._fn(self._x1)

    def test_metadata(self):
        assert list(self._t0.shape) == [33 * 8]
        assert list(self._t1.shape) == [33 * 8]
        assert self._t0.dtype == tf.uint8
        assert self._t1.dtype == tf.uint8

    def test_values(self):
        # there's at least one push in each bytecode (not all data is null)
        self.assertGreater(tf.reduce_sum(tf.cast(tf.not_equal(self._t0, 0), tf.int32)), 8)
        self.assertGreater(tf.reduce_sum(tf.cast(tf.not_equal(self._t1, 0), tf.int32)), 8)
        # most data is null
        self.assertGreater(tf.reduce_sum(tf.cast(tf.equal(self._t0, 0), tf.int32)), 32 * 4)
        self.assertGreater(tf.reduce_sum(tf.cast(tf.equal(self._t1, 0), tf.int32)), 32 * 4)
        # check the first opcode
        self.assertEqual(self._t0[0], self._b0[0])
        self.assertEqual(self._t1[0], self._b1[0])
        # length of the data in the first push
        __len0 = self._b0[0] - 0x5f # PUSH15
        __len1 = self._b1[0] - 0x5f # PUSH22
        # check the padding of the first push
        self.assertAllEqual(self._t0[1:33 - __len0], tf.zeros((32 - __len0,)))
        self.assertAllEqual(self._t1[1:33 - __len1], tf.zeros((32 - __len1,)))
        # check the data of the first push
        self.assertEqual(tf.reduce_sum(tf.cast(tf.equal(self._t0[33 - __len0:33], 0), tf.int32)), 0)
        self.assertEqual(tf.reduce_sum(tf.cast(tf.equal(self._t1[33 - __len1:33], 0), tf.int32)), 0)
        # the last instruction is not a push => null data
        self.assertAllEqual(self._t0[-32:], tf.zeros((32,)))
        self.assertAllEqual(self._t1[-32:], tf.zeros((32,)))

# PREPROCESSING ###############################################################

class PreprocessTest(tf.test.TestCase):
    def setUp(self):
        super(PreprocessTest, self).setUp()
        # preprocessing config
        self._config_categorical = {'batch_dim': 4, 'sample_dim': 33 * 128, 'token_dim': 33 * 2, 'output_dim': 256, 'padding_weight': 0.0, 'sample_weights': True, 'binary': False}
        self._config_binary = {'batch_dim': 4, 'sample_dim': 33 * 128, 'token_dim': 33 * 2, 'output_dim': 8, 'padding_weight': 0.0, 'sample_weights': True, 'binary': True}
        # specialized preprocessing fn
        self._preprocess_categorical = functools.partial(revml.contract.decoder.pipeline.preprocess, **self._config_categorical)
        self._preprocess_binary = functools.partial(revml.contract.decoder.pipeline.preprocess, **self._config_binary)
        # original dataset
        __b0 = b'3d602d80600a3d3981f3363d3d373d3d3d363d7339778bc77bd7a9456655b19fd4c5d0bf2071104e5af43d82803e903d91602b57fd5bf3'
        __b1 = b'7f602036038060203d373d3d3d923d343d355af13d82803e903d91601e57fd5bf33d5260203df3'
        __b2 = b'60a060405234801561001057600080fd5b506040516101093803806101098339818101604052602081101561003357600080fd5b50516001600160601b031960609190911b16608052600080546001600160a01b0319163317905560805160601c609561007460003980601b525060956000f3fe608060405236600a57005b348015601557600080fd5b506040517f0000000000000000000000000000000000000000000000000000000000000000903680600083376000808284865af43d9150816000843e808015605b578284f35b8284fdfea2646970667358221220c4a46717f67197616503d87ff2612122c70a6a9ed6d1f1999e000b68e428470964736f6c634300060c003300000000000000000000000080d0f44d6c1563de6ba356aa8dfe7abdbe8a174a'
        __b3 = b'608060405234801561001057600080fd5b506040516101e63803806101e68339818101604052602081101561003357600080fd5b8101908080519060200190929190505050600073ffffffffffffffffffffffffffffffffffffffff168173ffffffffffffffffffffffffffffffffffffffff1614156100ca576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004018080602001828103825260228152602001806101c46022913960400191505060405180910390fd5b806000806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505060ab806101196000396000f3fe608060405273ffffffffffffffffffffffffffffffffffffffff600054167fa619486e0000000000000000000000000000000000000000000000000000000060003514156050578060005260206000f35b3660008037600080366000845af43d6000803e60008114156070573d6000fd5b3d6000f3fea2646970667358221220d1429297349653a4918076d650332de1a1068c5f3e07c5c82360c277770b955264736f6c63430007060033496e76616c69642073696e676c65746f6e20616464726573732070726f7669646564000000000000000000000000d9db270c1b5e3bd161e8c8503c55ceabee709552'
        __b4 = b'900e60ffbb1010136ef09f3bf7356866882f5b9b56e9fbdbb373d44480469538'
        __b5 = b'2199ef7a217e1a55c1dfff22eb3a60af6fcdc91126d1dc7cbd6711c50a624cc0'
        __b6 = b'ce83d7b0b2a7fd96dc7ca8bf02d85b6b9acd71641f20080326d37421c296e9e2'
        __b7 = b'60008060008034415af1'
        self._dataset_before = tf.data.Dataset.from_tensor_slices([__b0, __b1, __b2, __b3, __b4, __b5, __b6, __b7,])
        # preprocessed datasets
        self._dataset_categorical = self._dataset_before.batch(self._config_categorical['batch_dim']).map(self._preprocess_categorical)
        self._dataset_binary = self._dataset_before.batch(self._config_categorical['batch_dim']).map(self._preprocess_binary)

    def test_specs(self):
        __inputs_spec, __targets_spec, __weights_spec = self._dataset_categorical.element_spec
        self.assertEqual(__inputs_spec.shape, (self._config_categorical['batch_dim'], self._config_categorical['sample_dim']))
        self.assertEqual(__targets_spec.shape, (self._config_categorical['batch_dim'], self._config_categorical['sample_dim'], self._config_categorical['output_dim']))
        self.assertEqual(__weights_spec.shape, (self._config_categorical['batch_dim'], self._config_categorical['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.int32)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__weights_spec.dtype, tf.dtypes.float32)
        __inputs_spec, __targets_spec, __weights_spec = self._dataset_binary.element_spec
        self.assertEqual(__inputs_spec.shape, (self._config_binary['batch_dim'], self._config_binary['sample_dim']))
        self.assertEqual(__targets_spec.shape, (self._config_binary['batch_dim'], self._config_binary['sample_dim'], self._config_binary['output_dim']))
        self.assertEqual(__weights_spec.shape, (self._config_binary['batch_dim'], self._config_binary['sample_dim']))
        self.assertEqual(__inputs_spec.dtype, tf.dtypes.int32)
        self.assertEqual(__targets_spec.dtype, tf.dtypes.float32)
        self.assertEqual(__weights_spec.dtype, tf.dtypes.float32)

    def test_values(self):
        # categorical
        __batch = iter(self._dataset_categorical)
        for _ in range(2):
            __x, __y, __m = next(__batch)
            __y = mlable.sampling.categorical(__y)
            self.assertAllEqual(__x[:, :self._config_categorical['token_dim']], tf.zeros((self._config_categorical['batch_dim'], self._config_categorical['token_dim'])))
            self.assertAllEqual(__x[:, self._config_categorical['token_dim']:2 * self._config_categorical['token_dim']], __y[:, :self._config_categorical['token_dim']])
        # binary
        __batch = iter(self._dataset_binary)
        for _ in range(2):
            __x, __y, __m = next(__batch)
            __y = mlable.sampling.binary(__y)
            self.assertAllEqual(__x[:, :self._config_categorical['token_dim']], tf.zeros((self._config_categorical['batch_dim'], self._config_categorical['token_dim'])))
            self.assertAllEqual(__x[:, self._config_categorical['token_dim']:2 * self._config_categorical['token_dim']], __y[:, :self._config_categorical['token_dim']])

    def test_weights(self):
        # categorical
        __batch = iter(self._dataset_categorical)
        for _ in range(2):
            __x, __y, __m = next(__batch)
            __y = mlable.sampling.categorical(__y)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __m = tf.cast(__m, dtype=tf.dtypes.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__x * __m, __x)
        # binary
        __batch = iter(self._dataset_binary)
        for _ in range(2):
            __x, __y, __m = next(__batch)
            __y = mlable.sampling.binary(__y)
            __x = tf.cast(__x, dtype=tf.dtypes.int32)
            __m = tf.cast(__m, dtype=tf.dtypes.int32)
            assert 0 < tf.reduce_sum(__m).numpy()
            assert tf.reduce_sum(__m).numpy() < tf.size(__m).numpy()
            self.assertAllEqual(__x * __m, __x)