import pytest
import tensorflow as tf
from tensorflow import keras
import tempfile

import utils


class BatchMatMulLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, MatrixA, MatrixB):
        return tf.matmul(MatrixA, MatrixB)

class BatchMatMulModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matmul_ = BatchMatMulLayer()

    def call(self, inputs, training=False, mask=None):
        o = self.matmul_(inputs[0], inputs[1])
        return o

@pytest.mark.parametrize("qtype", [True, False])
@pytest.mark.parametrize("m",     [3, 15])
@pytest.mark.parametrize("k",     [2, 1])
@pytest.mark.parametrize("n",     [4, 15])
@pytest.mark.parametrize("b",     [1])
def test_BatchMatMul(delegate_lib, qtype, m, k, n, b):
    a_shape = [b, m, k]
    b_shape = [b, k, n]
    model = BatchMatMulModel()
    model.build(input_shape=[a_shape, b_shape])

    fake_a = tf.random.normal(a_shape, 0, 127, tf.float32)
    fake_b = tf.random.normal(b_shape, 0, 127, tf.float32)
    model.predict([fake_a, fake_b], batch_size=b)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def data_set():
        for _ in range(10):
            yield [tf.random.normal(a_shape, 0, 127, tf.float32),
                   tf.random.normal(b_shape, 0, 127, tf.float32)]
    if (qtype is True):
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = data_set
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    fp = tempfile.NamedTemporaryFile()
    tflite_model = converter.convert()
    fp.write(tflite_model)
    fp.flush()

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(fp.name)
    npu_out = npu_.run(fp.name, gold_in)
    fp.close()
    pytest.approx(gold_out,npu_out)
