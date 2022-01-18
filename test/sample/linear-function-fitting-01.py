import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
线性函数 y=0.1x + 0.2
'''
# 使用numpy生成100个0-1的随机点作为x
x_data = np.random.rand(100)
# 生成一些随机扰动
noise = np.random.normal(0, 0.01, x_data.shape)
# 构建目标值，符合线性分布
y_data = x_data * 0.1 + 0.2 + noise
# 画散点图
plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = tf.keras.Sequential()
# Dense为全连接层
# 在模型中添加一个全连接层
# units为输出神经元个数，input_dim为输入神经元个数
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# 设置模型的优化器和迭代函数，学习率为0.03
# sgd:Stochastic gradient descent，随机梯度下降法
# mse:Mean Squared Error，均方误差
model.compile(optimizer=tf.keras.optimizers.SGD(0.03), loss='mse')

# 训练3000个批次
for step in range(3001):
    # 训练一个批次数据，返回loss值
    loss = model.train_on_batch(x_data, y_data)
    # 每 1000 个 batch 打印一次 loss 值
    if step % 1000 == 0:
        print('loss: ', loss)
        # 定义一个 2*2 的图，当前是第 i/1000+1 个图
        plt.subplot(2, 2, int(step / 1000 + 1))
        # 使用 predict 对数据进行预测，得到预测值 y_pred
        y_pred = model.predict(x_data)
        # 显示随机点
        plt.scatter(x_data, y_data)
        plt.plot(x_data, y_pred, 'r-', lw=3)
        # 不显示坐标
        plt.axis('off')
        # 图片的标题设置
        plt.title("picture:" + str(int(step / 1000 + 1)))

plt.show()
