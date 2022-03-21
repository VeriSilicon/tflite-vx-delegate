import pytest
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import utils
import tempfile

@pytest.mark.parametrize("batch_shape, steps, input_dim",  [(1,4,2)])
@pytest.mark.parametrize("filters",  [2])
@pytest.mark.parametrize("k_size",  [4])
@pytest.mark.parametrize("strides",  [1])
@pytest.mark.parametrize("groups",  [2])
@pytest.mark.parametrize("padding",  ['valid'])
@pytest.mark.parametrize("bias_initializer",  ['zeros','ones'])
@pytest.mark.parametrize("qtype",   [True,False])

def test_conv1d(delegate_lib, batch_shape, steps, input_dim, filters, k_size, strides, groups, padding, bias_initializer, qtype):
    input_shape = (batch_shape, steps, input_dim)
    kernel_size = k_size
    input_dtype = tf.float32
    fake_input = tf.random.normal(input_shape, 0, 127, input_dtype)

    def rand_dataset():
        for _ in range(100):
            yield [tf.random.normal(input_shape, 0, 127, input_dtype)]

    inputs = keras.Input(shape = input_shape[1:], batch_size= input_shape[0], name= "input")
    conv1d = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups, bias_initializer=bias_initializer)(inputs)
    model = keras.Model(inputs = inputs, outputs = conv1d)

    model.build(input_shape)
    model.summary()

    model.predict([fake_input])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if (qtype is True):
        converter.representative_dataset = rand_dataset
        converter.inference_input_type = tf.int8
        converter.inference_input_type = tf.int8
    tflite_model = converter.convert()

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()

    fp = tempfile.NamedTemporaryFile()
    fp.write(tflite_model)
    fp.flush()
    (gold_in, gold_out)= cpu_.run_with_rand_data(fp.name)
    npu_out = npu_.run(fp.name, gold_in)
    fp.close()

    # model_path = "/tmp/model.tflite"
    # open(model_path, "wb").write(tflite_model)
    # (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    # npu_out = npu_.run(model_path, gold_in)

    for (g, n) in zip(gold_out, npu_out):
       assert pytest.approx(g, n[1])