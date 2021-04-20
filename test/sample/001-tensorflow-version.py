import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# 验证环境
print("TensorFlow version: " + tf.__version__)
print("Keras version: " + tf.keras.__version__)
print()

devices = tf.config.list_physical_devices()
for device in devices:
    print(devices)

# 创建两个常量
a = tf.constant([1, 2], name='a')
b = tf.constant([1, 2], name='b')

# 两个向量相加
print(tf.add(a, b))
