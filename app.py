import os

from flask import Flask


@app.route("/")
def hello_world():
    return "Hello World !"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
