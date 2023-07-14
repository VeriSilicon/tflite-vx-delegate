import pytest
import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import utils
import tempfile
import os

def test_layout_infer(delegate_lib, save_model):
    input_shape = [1, 1024, 768]

    input = tf.keras.layers.Input(shape=input_shape[1:], batch_size = 1)
    lyn_0_output = tf.keras.layers.LayerNormalization(axis = 1, name="Layernorm_0")(input)

    # ----> Case 0
    c1 = tf.random.normal((1, 1024, 768))
    in0 = keras.layers.Add()([lyn_0_output, c1])
    ln = tf.keras.layers.LayerNormalization(axis=1)(in0)
    lyn_1_output = keras.layers.Add()([in0, c1])

    add_out = keras.layers.Add()([ln, lyn_1_output])
    output = tf.keras.layers.LayerNormalization(axis=1)(add_out)
    # <----

    # # ----> Case 1
    # lyn_1_output = tf.keras.layers.LayerNormalization(axis=[1])(input)
    # mm_out = tf.keras.layers.Dot(axes=(1,1))([lyn_0_output[:,0:256], lyn_1_output[:,256:512]])
    # lyn_2_output = tf.keras.layers.LayerNormalization()(mm_out)
    # add_input2 = tf.random.normal((1,1))
    # output = tf.keras.layers.Add()([lyn_2_output, add_input2])
    # # <---

    # ----> Case 2
    # fc0 = tf.keras.layers.Dense(768)(lyn_0_output)
    # output= tf.keras.layers.Dense(768)(fc0)
    # <---- Case 2

    # ----> case 3: before GEMM
    # emb = tf.keras.layers.Dense(768)(lyn_0_output)
    # reshape = tf.keras.layers.Reshape((1024, 64, 4, 3))(emb)
    # permute = tf.keras.layers.Permute((4, 3, 1, 2))(reshape)
    # output = tf.keras.layers.Add()([permute[:,0:1,:,:,:], permute[:,1:2,:,:,:], permute[:,2:3,:,:,:]])
    # <---

    # -----> case : GEMM
    # output = tf.keras.layers.Dot(axes=(2,2))([input, input])

    # genenral for model
    model = keras.Model(inputs = input, outputs = output)

    model.build(input_shape = input_shape)
    model.summary()

    def rand_dataset():
            for _ in range(10):
                yield [ tf.random.normal(input_shape, 0, 127, tf.float32) ]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rand_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    model_name = "layer_infer.tflite"
    tflite_model = converter.convert()
    if (os.path.exists(save_model)):
        model_path = save_model + "/" + model_name
        print("echo: save model to ", model_path)
        open(model_path, "wb").write(tflite_model)

    cpu_ = utils.cpu()
    (gold_in, gold_out)= cpu_.run_with_rand_data(model_path)
    pass