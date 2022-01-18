import logging
import os

from definitions import ROOT_DIR

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

data_path = os.path.join(ROOT_DIR, 'assets/mnist/data_set/mnist.npz')
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data_path)
print(train_images.shape)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])

plt.show()
