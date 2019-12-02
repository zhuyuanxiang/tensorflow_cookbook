# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0206_back_propagation.py
@Version    :   v0.1
@Time       :   2019-10-29 15:33
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0206，P30
@Desc       :   TensorFlow 进阶，TensorFlow 实现反向传播
"""
# common imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from tensorflow.python.framework import ops

from tools import show_title

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)

# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)

# 初始化默认的计算图
ops.reset_default_graph()
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Open graph session
sess = tf.Session()


def regression_example():
    show_title("反向传播算法的回归应用")
    # 创建数据集
    # x-data: 从正态分布 N(1, 0.1) 中抽取100个数据
    # target: 100个数据的对应目标值
    # x - data * A = target，理论上 A = 10.
    # 输入数据，因为是依据正态分布随机生成的数据，所以结果不可能得到完全一样的斜率
    x_vals = np.random.normal(1, 0.1, 100)
    # 预测数据，不同的预测数据，参数 A 会得到不同的结果
    # ToSee：注意下面三种目标设置导致的不同
    y_vals = np.repeat(10., 100)
    # y_vals = 3 + x_vals
    # y_vals = 3 * x_vals + 0.5
    plt.scatter(x_vals, y_vals)
    plt.title("原始数据的散点图")

    x_data = tf.placeholder(shape = [1], dtype = tf.float32)
    y_target = tf.placeholder(shape = [1], dtype = tf.float32)

    # A 是模型的参数（定义为变量）
    A = tf.Variable(tf.random_normal(shape = [1]))

    # 定义操作到计算图中，以下两种方式是一样的
    my_output = tf.multiply(x_data, A)
    my_output = x_data * A

    # 使用 L2 损失函数
    loss = tf.square(my_output - y_target)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # 使用 梯度下降 优化器，目标是最小化损失
    learning_rate = 0.02
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss)

    # Run Loop
    for i in range(1000):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        feed_dict = {x_data: rand_x, y_target: rand_y}
        sess.run(train_step, feed_dict = feed_dict)
        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict = feed_dict)))


def classification_example():
    show_title("反向传播算法的分类问题应用")
    # 创建的数据集
    # x-data: 从 N(-1, 1) 中抽取50个随机样本 + 从 N(1, 1) 中抽取50个随机样本
    # target: 前面 50 个样本对应标签为0，后面50个样本对应标签为1
    # 训练二分类模型
    # If sigmoid(x+A) < 0.5 -> 0 else 1，理论上： A = -(mean1 + mean2)/2
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
    plt.scatter(x_vals, y_vals)
    plt.title("原始数据的散点图")

    x_data = tf.placeholder(shape = [1], dtype = tf.float32)
    y_target = tf.placeholder(shape = [1], dtype = tf.float32)

    # Create variable (one model parameter = A)
    # A 是模型参数
    A = tf.Variable(tf.random_normal(mean = 10, shape = [1]))

    # Add operation to graph
    # Want to create the operation sigmoid(x + A)
    # Note, the sigmoid() part is in the loss function
    # sigmoid()在代价函数中定义
    # 注：这个例子中定义 sigmoid() 的方法不好，代码不清晰，特别是对于相对复杂的代码来说，阅读起来有困难
    # 保留在这里不做修改，可以思考如何定义更好
    my_output = x_data + A
    # Now we have to add another dimension to each (batch size of 1)
    # 因为指定的损失函数期望批量数据，所以增加一个维度用于定义批量的大小（批量大小默认为1）
    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y_target, 0)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Add classification loss (cross entropy)
    # 原代码把输入数据和目标数据放反了。
    # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = my_output_expanded, logits = y_target_expanded)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = my_output_expanded, labels = y_target_expanded)
    # ToDo: 可以尝试更换其他代价函数
    # xentropy = - (y_target_expanded @ tf.log(my_output_expanded)) \
    #            - ((1. - y_target_expanded) @ tf.log(1. - my_output_expanded))
    # xentropy = -(y_target * tf.log(my_output)) - ((1. - y_target) * tf.log(1. - my_output))
    # xentropy = tf.maximum(0., 1. - tf.multiply(y_target_expanded, my_output_expanded))
    # xentropy= tf.maximum(0., 1. - tf.multiply(y_vals, x_vals))

    # Create Optimizer
    # 使用梯度下降 优化器
    learning_rate = 0.05
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(xentropy)

    # Run loop
    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
        if (i + 1) % 200 == 0:
            print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(xentropy, feed_dict = {
                    x_data: rand_x, y_target: rand_y
            })))

    # Evaluate Predictions
    predictions = []
    for i in range(len(x_vals)):
        x_val = [x_vals[i]]
        prediction = sess.run(tf.round(tf.sigmoid(my_output)), feed_dict = {x_data: x_val})
        predictions.append(prediction[0])

    accuracy = sum(x == y for x, y in zip(predictions, y_vals)) / 100.
    print('评估精度 = ' + str(np.round(accuracy, 2)))


# -----------------------------------------------------------------
if __name__ == "__main__":
    # regression_example()
    classification_example()

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
