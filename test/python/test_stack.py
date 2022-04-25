import pytest
import tensorflow as tf
from tensorflow.python import keras
import tempfile

import utils

input = tf.random.normal([1,4,3,2], 0, 4, tf.float32)   #nhwc
kernel = tf.random.normal([3,3,2,3], 0, 4, tf.float32)   #hwio
class Conv2dLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @tf.function
    def __call__(self, x):
        return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')

class StackLayer(keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def __call__(self, input):
        return tf.stack([input, input],axis = self.axis)

class Conv2dStackModel(keras.Model):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_ = Conv2dLayer()
        self.stack_ = StackLayer(axis)

    # @tf.function
    def call(self, input):
        conv2d_out = self.conv2d_(input)  #as only one input, don't use input[0],input[1]
        o = self.stack_(conv2d_out)
        return o

@pytest.mark.parametrize("qtype",         [False])
@pytest.mark.parametrize("axis",          [0,1,2,3,4])
def test_stride_slice(delegate_lib, axis, qtype):

    model = Conv2dStackModel(axis)
    model.build(input_shape=input.shape)  #while multiply input, use [x.shape, y.shape]
    model.predict(input)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def data_set():
        for _ in range(10):
            yield [tf.random.normal(input.shape, 0, 127, tf.float32),
                   tf.random.normal(kernel.shape, 0, 127, tf.float32)]
    if (qtype is True):
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = data_set
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()

    # model_path = "/tmp/model.tflite"
    # open(model_path, "wb").write(tflite_model)
    # (gold_in, gold_out)= cpu_.run_with_rand_data(tflite_model)
    # npu_out = npu_.run(tflite_model, gold_in)

    fp = tempfile.NamedTemporaryFile()
    fp.write(tflite_model)
    fp.flush()
    (gold_in, gold_out)= cpu_.run_with_rand_data(fp.name)
    npu_out = npu_.run(fp.name, gold_in)
    fp.close()

    for (g, n) in zip(gold_out, npu_out):
        assert pytest.approx(g, n)
