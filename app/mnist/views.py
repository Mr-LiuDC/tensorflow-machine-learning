from flask import render_template

from . import mnist


@mnist.router('/')
def homepage():
    return render_template('mnist/mnist-index.html', title="Welcome")
