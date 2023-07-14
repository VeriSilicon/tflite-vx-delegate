import pytest
import tensorflow as tf
import numpy as np
import utils

@pytest.mark.parametrize("batch_size",  [1])
@pytest.mark.parametrize("in_w, in_h, in_d, k_w, k_h, k_d", [(4, 4, 4, 3, 3, 2), (112, 112, 56, 3, 3, 2)])
@pytest.mark.parametrize("in_ch",       [1])
@pytest.mark.parametrize("out_ch",      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("qtype",       [True, False])
def test_conv3d(delegate_lib, batch_size, in_w, in_h, in_d, in_ch, out_ch, k_w, k_h, k_d, qtype):
    # input layout [N, H, W, D, C]
    input_shape = [batch_size, in_h, in_w, in_d, in_ch]
    out_channel = out_ch
    # kernel layout [Kd, Kh, Kw]
    kernel_shape = [1, 2, 2]
    input_dtype = tf.float32

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = input_shape[1:], batch_size= input_shape[0]),
        tf.keras.layers.Conv3D(filters = out_channel, kernel_size= kernel_shape)
        ])
    model.build()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    def rand_calibration():
        for _ in range(100):
            yield [tf.random.normal(input_shape[0:], 0, 127, input_dtype)]

    if (qtype is True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rand_calibration
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    model_path = "./test_model.tflite"
    open(model_path, "wb").write(tflite_model)

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    npu_out = npu_.run(model_path, gold_in)

    for (g, n) in zip(gold_out, npu_out):
        assert pytest.approx(g, n)
