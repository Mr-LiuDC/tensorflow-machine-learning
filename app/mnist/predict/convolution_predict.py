import os

import numpy as np
import tensorflow as tf
from PIL import Image

from app.mnist.training.convolution_training import CNN
from definitions import ROOT_DIR


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('../ckpt/convolution')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('-------> Predict digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict(os.path.join(ROOT_DIR, './test_images/0.png'))
    app.predict('./test_images/1.png')
    app.predict('./test_images/4.png')
