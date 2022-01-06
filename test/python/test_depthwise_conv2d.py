import pytest
import tensorflow as tf
import numpy as np
import utils
import tempfile

@pytest.mark.parametrize("batch_size, channels",  [(1,2),(2,4)])
@pytest.mark.parametrize("rows, cols",  [(224,224)])
@pytest.mark.parametrize("multiplier",  [1,2,3])
@pytest.mark.parametrize("k_rows, k_cols",  [(2,2),(14,14),(20,20)])
@pytest.mark.parametrize("strides",  [1,2])
@pytest.mark.parametrize("padding",  ['valid','same'])
@pytest.mark.parametrize("qtype",   [True,False])
def test_depthwise_conv2d(delegate_lib, batch_size, channels, rows, cols, multiplier, k_rows, k_cols, strides, padding, qtype):
    input_shape = (batch_size, rows, cols, channels)
    kernel_size = (k_rows, k_cols)
    input_dtype = tf.float32
    fake_input = tf.random.normal(input_shape, 0, 127, input_dtype)

    def rand_dataset():
        for _ in range(100):
            yield [tf.random.normal(input_shape, 0, 127, input_dtype)]

    inputs = tf.keras.Input(shape = input_shape[1:], batch_size= input_shape[0], name= "input")
    depthwise_conv2d = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding,
                        depth_multiplier=multiplier, name="Mydepthconv2d")(inputs)
    model = tf.keras.Model(inputs = inputs, outputs = depthwise_conv2d)

    model.build(input_shape)
    model.summary()

    model.predict([fake_input], batch_size=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if (qtype is True):
        converter.representative_dataset = rand_dataset
        converter.inference_input_type = tf.int8
        converter.inference_input_type = tf.int8

    fp = tempfile.NamedTemporaryFile()
    tflite_model = converter.convert()
    fp.write(tflite_model)
    fp.flush()

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(fp.name)
    npu_out = npu_.run(fp.name, gold_in)
    fp.close()
    for (g, n) in zip(gold_out, npu_out):
       assert pytest.approx(g, n[1])