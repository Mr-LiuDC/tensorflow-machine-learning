import tensorflow as tf

print(tf.__version__)
print(tf.keras.__version__)
print(tf.test.is_gpu_available)
print(tf.test.gpu_device_name)
print(tf.config.list_physical_devices('GPU'))

# 创建两个常量
a = tf.constant([1, 2], name='a')
b = tf.constant([1, 2], name='b')

# 两个向量相加
print(tf.add(a, b))
