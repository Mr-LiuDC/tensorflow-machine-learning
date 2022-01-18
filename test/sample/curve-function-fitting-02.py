import os

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_dim=1, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='tanh'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.3), loss='mse')

for step in range(2501):
    loss = model.train_on_batch(x, y)
    if step % 50 == 0:
        print("Step={0}, Loss={1}".format(step, loss))
        p_y = model.predict(x)
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, p_y, 'r-', lw=5)
        plt.axis('off')
        plt.title("Step={0}, Loss={1}".format(step, loss), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.01)

plt.show()
