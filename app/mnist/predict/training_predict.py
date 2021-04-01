import os

import numpy as np
import tensorflow as tf
from PIL import Image

from app.mnist.training.convolution_training import CNN
from app.mnist.training.regression_training import RGN
from definitions import ROOT_DIR


class Predict(object):
    def __init__(self):
        self.rgn = RGN()
        self.cnn = CNN()
        rgn_latest = tf.train.latest_checkpoint('../ckpt/regression')
        cnn_latest = tf.train.latest_checkpoint('../ckpt/convolution')
        # 恢复网络权重
        self.rgn.model.load_weights(rgn_latest)
        self.cnn.model.load_weights(cnn_latest)

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.0
        x = np.array([1 - img])

        # API refer: https://keras.io/models/model/
        y_1 = self.rgn.model.predict(x)
        y_2 = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y_1[0])
        print(y_2[0])
        print('-------> RGN predict digit', np.argmax(y_1[0]))
        print('-------> CNN predict digit', np.argmax(y_2[0]))


if __name__ == "__main__":
    predict = Predict()
    predict.predict(os.path.join(ROOT_DIR, 'assets/mnist/test_set/test_image_single_28_001.jpg'))
    predict.predict(os.path.join(ROOT_DIR, 'assets/mnist/test_set/test_image_single_28_002.jpg'))
    predict.predict(os.path.join(ROOT_DIR, 'assets/mnist/test_set/test_image_single_28_003.jpg'))
    predict.predict(os.path.join(ROOT_DIR, 'assets/mnist/test_set/test_image_single_28_004.jpg'))
    predict.predict(os.path.join(ROOT_DIR, 'assets/mnist/test_set/test_image_single_28_005.jpg'))
