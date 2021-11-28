import tensorflow as tf
import numpy as np
from tensorflow.python import keras
tf.random.set_seed(4)

x = tf.random.normal([1, 3, 2], 0, 10, tf.float32)
y = tf.random.normal([1, 2, 4], 0, 10, tf.float32)

class BatchMatMulLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @tf.function
    def __call__(self, MatrixA, MatrixB):
        return tf.matmul(MatrixA, MatrixB)

class BatchMatMulModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matmul_ = BatchMatMulLayer()

    # @tf.function
    def call(self, inputs, training=False, mask=None):
        o = self.matmul_(inputs[0], inputs[1])
        return o

model = BatchMatMulModel()
model.build(input_shape=[x.shape, y.shape])
model.summary()
output = model.predict([x, y], verbose=1)
print(output)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

x = x.numpy()
print(x)
y = y.numpy()
print(y)
x.tofile("tf_data/x_bin")
y.tofile("tf_data/y_bin")
open("BatchMatmul.tflite","wb").write(tflite_quant_model)
