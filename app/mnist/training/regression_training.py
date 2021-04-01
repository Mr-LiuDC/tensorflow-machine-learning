import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from definitions import ROOT_DIR


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

data_path = os.path.join(ROOT_DIR, 'assets/mnist/data_set/mnist.npz')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(data_path)
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
check_path = '../ckpt/regression/cp-{epoch:04d}.ckpt'
log_path = os.path.join(ROOT_DIR, 'logs/regression_training_logs')
save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, save_model_cb])

plot_learning_curves(history)

show_single_image(x_test[0])
print("模型的预测结果是：", predict_data(x_test[0]))
