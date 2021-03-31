import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def show_single_image(img_arr):
    img_arr = img_arr.reshape(28, 28)
    plt.imshow(img_arr, cmap="binary")
    plt.show()


def plot_learning_curves(parameters):
    pd.DataFrame(parameters.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)
    plt.show()


def predict_data(test_data):
    pred = model.predict(test_data.reshape(-1, 28, 28, 1))
    return np.argmax(pred)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print("训练集的图片数据维度：", x_train.shape)
print("训练集的标签数据维度：", y_train.shape)
print("测试集的图片数据维度：", x_test.shape)
print("测试集的标签数据维度：", y_test.shape)

x = 0
show_single_image(x_train[x])
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
tf.summary.create_file_writer("../../logs")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

plot_learning_curves(history)

show_single_image(x_test[0])
print("模型的预测结果是：", predict_data(x_test[0]))
model.save('../trained_models/my_mnist_trained_model.h5')
