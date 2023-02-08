import pytest
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import utils
import tempfile
import os

@pytest.mark.parametrize("num_of_seq", [197])
@pytest.mark.parametrize("seq_len", [768])
# @pytest.mark.parametrize("in_num_heads", [12, 24, 64])
@pytest.mark.parametrize("in_num_heads", [12])
@pytest.mark.parametrize("in_key_dim", [64])
@pytest.mark.parametrize("qtype",       [True])
@pytest.mark.parametrize("enable_mask", [True])
def test_attention(delegate_lib, save_model, num_of_seq, seq_len, in_num_heads, in_key_dim, qtype, enable_mask):
    input_shape = (num_of_seq, seq_len)
    input = tf.keras.Input(shape=input_shape)
    attention_mask = tf.keras.Input((1, num_of_seq, num_of_seq))
    if (enable_mask == True):
        output = tf.keras.layers.MultiHeadAttention(num_heads=in_num_heads, key_dim=in_key_dim, attention_axes=(1))(input, input, attention_mask = attention_mask)
    else :
        output = tf.keras.layers.MultiHeadAttention(num_heads=in_num_heads, key_dim=in_key_dim, attention_axes=(1))(input, input)

    model = keras.Model(inputs = (input, attention_mask), outputs = output)

    model.build(input_shape=input_shape)
    model.summary()

    def rand_dataset():
            for _ in range(10):
                yield [tf.random.normal(input_shape, 0, 127, tf.float32), tf.ones((1,num_of_seq, num_of_seq))]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    if (qtype is True):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rand_dataset
        converter.inference_input_type = tf.int8
        converter.inference_input_type = tf.int8

    tflite_model = converter.convert()

    model_path = ""
    temp_model = tempfile.NamedTemporaryFile()
    model_name = "attention_nseq{}.seq_len.{}.heads.{}.key_dim.{}.qtype.{}.mask.{}.tflite".format(num_of_seq, seq_len, in_num_heads,in_key_dim,qtype,enable_mask)
    if (os.path.exists(save_model)):
        model_path = save_model + "/" + model_name
        print("echo: save model to ", model_path)
        open(model_path, "wb").write(tflite_model)
    else:
        print("Debug ECHO: save model to temp file(give patch{} not exist".format(save_model))
        temp_model.write(tflite_model)
        model_path = temp_model.name

    npu_ = utils.npu(delegate_lib)
    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    npu_out = npu_.run(model_path, gold_in)
    for (g, n) in zip(gold_out, npu_out):
        assert g == pytest.approx(n[1])
    temp_model.close()
