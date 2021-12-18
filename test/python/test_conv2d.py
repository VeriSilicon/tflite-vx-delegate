import pytest
import tensorflow as tf
import numpy as np
import utils

@pytest.mark.parametrize("batch_size",  [1])
@pytest.mark.parametrize("in_w, in_h, k_w, k_h", [(4,4,3,3), (224, 224, 3, 3)])
@pytest.mark.parametrize("in_ch",       [1])
@pytest.mark.parametrize("out_ch",      [1, 4])
@pytest.mark.parametrize("qtype",       [True, False])
def test_conv2d(delegate_lib, batch_size, in_w, in_h, in_ch, out_ch, k_w, k_h, qtype):
    input_shape = [batch_size, in_w, in_h, in_ch]
    out_channel = out_ch
    kernel_shape = [k_w, k_h]
    input_dtype = tf.float32

    def rand_calibration():
        for _ in range(100):
            yield [tf.random.normal((in_w, in_h, in_ch), 0, 127, input_dtype)]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = input_shape[1:], batch_size= input_shape[0]),
        tf.keras.layers.Conv2D(filters = out_channel, kernel_size= kernel_shape)
        ])
    model.build()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if (input_dtype is True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rand_calibration
        converter.inference_input_type = tf.int8
        converter.inference_input_type = tf.int8
    
    tflite_model = converter.convert()       
    model_path = "/tmp/test_model.tflite"
    open(model_path, "wb").write(tflite_model)

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    npu_out = npu_.run(model_path, gold_in)

    for (g, n) in zip(gold_out, npu_out):
        assert pytest.approx(g, n[1])
