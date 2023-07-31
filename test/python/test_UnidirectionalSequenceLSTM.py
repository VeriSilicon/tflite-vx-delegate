import pytest
import tensorflow as tf
from tensorflow import keras
import utils

@pytest.mark.parametrize("batch,timesteps,feature",  [(32,10,8),(5,28,28)])
@pytest.mark.parametrize("unit",          [4])
@pytest.mark.parametrize("unroll_type",   [False])

def test_UnidirectionalSequenceLSTM(delegate_lib, batch, timesteps, feature, unit, unroll_type):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape = (timesteps,feature), batch_size=batch))
    model.add(keras.layers.LSTM(units = unit,unroll = unroll_type))
    model.build()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    model_path = "./test_model.tflite"
    open(model_path, "wb").write(tflite_model)

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    npu_out = npu_.run(model_path, gold_in)

    pytest.approx(gold_out,npu_out)
