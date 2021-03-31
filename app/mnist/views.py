from flask import render_template

from . import mnist


@mnist.route('/')
def mnist_page():
    return render_template('mnist/mnist-index.html', title="MNIST")


@mnist.route("/api/mnist", methods=['POST'])
def mnist_mnist():
    return ""
