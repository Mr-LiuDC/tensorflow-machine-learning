import os
import logging

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import datasets

from definitions import ROOT_DIR

'''
采用线性回归训练
'''


class RGN(object):
    def __init__(self):
        model = tf.keras.models.Sequential([
            # 随机丢弃 20 % 神经元, 防止过度拟合
            tf.keras.layers.Dropout(0.2),
            # 二维数组（28 x 28 像素）拉平成一维数组
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            # 128个神经元并与上层728个神经元相互连接
            tf.keras.layers.Dense(128, activation='relu'),
            # 10个神经元，并使用softmax作为激活函数，这10个神经元的输出就是最终结的结果
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model = model


class DataSource(object):
    def __init__(self):
        data_path = os.path.join(ROOT_DIR, 'assets/mnist/data_set/mnist.npz')
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 数据归一化
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


class Train:
    def __init__(self):
        self.rgn = RGN()
        self.data = DataSource()

    def train(self):
        check_path = '../ckpt/regression/cp-{epoch:04d}.ckpt'
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)
        self.rgn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        log_path = os.path.join(ROOT_DIR, 'logs/regression_training_logs')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
        self.rgn.model.fit(self.data.train_images, self.data.train_labels, epochs=5,
                           validation_data=(self.data.train_images, self.data.train_labels),
                           callbacks=[tensorboard_callback, save_model_cb])
        test_loss, test_acc = self.rgn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()
