import pytest
import tensorflow as tf
from tensorflow import keras
import utils

@pytest.mark.parametrize("num_units",  [2])
@pytest.mark.parametrize("feature",  [4])

def test_GRUCell(delegate_lib, num_units, feature):
    input_shape = (1, feature)
    h_shape = (1, num_units)
    x = tf.constant([1,2,3,4])
    # initialize h_state tensor
    h = [tf.zeros(h_shape)]

    input1 = keras.Input(shape = input_shape[1:], batch_size= input_shape[0], name= "input")
    input2 = keras.Input(shape = h_shape[1:], batch_size= h_shape[0], name= "h")
    grucell = tf.keras.layers.GRUCell(num_units)(input1,input2) # multiple inputs

    model = keras.Model(inputs = [input1,input2], outputs = grucell)

    model.build([input_shape, h_shape])
    model.summary()

    model.predict([x,h])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()

    # fp = tempfile.NamedTemporaryFile()
    # fp.write(tflite_model)
    # fp.flush()
    # (gold_in, gold_out)= cpu_.run_with_rand_data(fp.name)
    # npu_out = npu_.run(fp.name, gold_in)
    # fp.close()

    model_path = "/tmp/model.tflite"
    open(model_path, "wb").write(tflite_model)
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)

    npu_out = npu_.run(model_path, gold_in)

    pytest.approx(gold_out,npu_out)