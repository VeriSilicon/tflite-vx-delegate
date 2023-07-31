import pytest
import tensorflow as tf
from tensorflow import keras
import utils
import tempfile

@pytest.mark.parametrize("batch_size, channels",  [(1,1),(2,2)])
@pytest.mark.parametrize("rows, cols",  [(224,224)])
@pytest.mark.parametrize("filters",  [1,2])
@pytest.mark.parametrize("k_rows, k_cols",  [(3,3)])
@pytest.mark.parametrize("strides",  [1,2])
@pytest.mark.parametrize("padding",  ['valid','same'])
@pytest.mark.parametrize("bias_initializer",  ['zeros','ones'])
@pytest.mark.parametrize("qtype",   [True,False])

def test_transpose_conv2d(delegate_lib, batch_size, channels, filters, rows, cols, k_rows, k_cols, strides, padding, bias_initializer, qtype):
    input_shape = (batch_size, rows, cols, channels)
    kernel_size = (k_rows, k_cols)
    input_dtype = tf.float32
    fake_input = tf.random.normal(input_shape, 0, 127, input_dtype)

    def rand_dataset():
        for _ in range(100):
            yield [tf.random.normal(input_shape, 0, 127, input_dtype)]

    inputs = keras.Input(shape = input_shape[1:], batch_size= input_shape[0], name= "input")
    transpose_conv2d = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, bias_initializer=bias_initializer)(inputs)
    model = keras.Model(inputs = inputs, outputs = transpose_conv2d)

    model.build(input_shape)
    model.summary()

    model.predict([fake_input])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if (qtype is True):
        converter.representative_dataset = rand_dataset
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
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
    pytest.approx(gold_out,npu_out)