import tensorflow as tf
import numpy as np
from tensorflow.python import keras
tf.random.set_seed(4)

x = tf.random.normal([1, 3, 2], 0, 4, tf.float32)
y = tf.random.normal([1, 2, 4], 0, 4, tf.float32)

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

def data_set():
    # mutiply input yield here, need float dtype
    for _ in range( 100 ):
        yield [x, y]

model = BatchMatMulModel()
model.build(input_shape=[x.shape, y.shape])
model.summary()
output = model.predict([x, y], batch_size = 5, verbose=1)
print(output)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = data_set
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

x = x.numpy().astype(np.int8)
y = y.numpy().astype(np.int8)
x.tofile("tf_data/int8x_bin")
y.tofile("tf_data/int8y_bin")
open("int8BatchMatmul.tflite","wb").write(tflite_quant_model)