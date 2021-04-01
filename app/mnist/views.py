import numpy as np
import tensorflow as tf
from flask import render_template, Flask, request, json

from . import mnist
from .training.convolution_training import CNN
from .training.regression_training import RGN

app = Flask(__name__)
root_path = app.root_path


@mnist.route('/')
def mnist_page():
    return render_template('mnist/mnist-index.html', title="MNIST")


@mnist.route("/api/mnist", methods=['POST'])
def mnist_mnist():
    input_data = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)

    rgn_latest = tf.train.latest_checkpoint(root_path + '/ckpt/regression')
    cnn_latest = tf.train.latest_checkpoint(root_path + '/ckpt/convolution')
    rgn = RGN()
    cnn = CNN()
    # 恢复网络权重
    rgn.model.load_weights(rgn_latest)
    cnn.model.load_weights(cnn_latest)

    pred_1 = rgn.model.predict(input_data.reshape(-1, 28, 28, 1)).flatten().tolist()
    pred_2 = cnn.model.predict(input_data.reshape(-1, 28, 28, 1)).flatten().tolist()
    return json.jsonify(results=[pred_1, pred_2])
