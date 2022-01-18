import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
线性函数 y=0.1x + 0.2
'''
x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
model.compile(optimizer=tf.keras.optimizers.SGD(0.03), loss='mse')

for step in range(2001):
    loss = model.train_on_batch(x_data, y_data)
    if step % 50 == 0:
        print("Step={0}, Loss={1}".format(step, loss))
        p_y = model.predict(x_data)
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, p_y, 'r-', lw=5)
        plt.axis('off')
        plt.title("Step={0}, Loss={1}".format(step, loss), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.01)

plt.show()
