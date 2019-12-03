# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0208_combining_everything_together.py
@Version    :   v0.1
@Time       :   2019-10-30 10:11
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0208，P37
@Desc       :   TensorFlow 进阶，TensorFlow 创建分类器
"""
# common imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from sklearn.datasets import load_iris
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

show_title("TensorFlow 创建分类器")

# Load the iris data（150条数据）
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 实现一个二值分类器，预测一朵花是否为山鸢尾（setosa, iris.target==0, binary_target==1.0)
# 只使用两个特征：花瓣长度（pedal.length,x[2]）和花瓣宽度（pedal.width,x[3]）
iris = load_iris()
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# Declare batch size
batch_size = 20

# Declare placeholders
x1_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
x2_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
# ToDo: 二维处理没有成功。
# x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)

# Create variables A and b (0 = x1 - (x2 * A + b))
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Add model to graph:
# x1 - (x2 * A + b)
my_output = x1_data - ((x2_data @ A) + b)
# my_mult = tf.matmul(x2_data, A)
# my_add = tf.add(my_mult, b)
# my_output = tf.subtract(x1_data, my_add)
# my_output = tf.subtract(x1_data, tf.add(tf.matmul(x2_data, A), b))
# my_mult = tf.matmul(x_data[1], A)
# my_add = tf.add(my_mult, b)
# my_output = tf.subtract(x1_data[0], my_add)
# my_output = tf.subtract(x_data[0], tf.add(tf.matmul(x_data[1], A), b))

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target, logits = my_output)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Run Loop
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size = batch_size)

    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict = {x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})

    # rand_x = np.transpose([iris_2d[rand_index]])
    # rand_y = np.transpose([binary_target[rand_index]])
    # sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        pass
    pass

# Visualize Results
# Pull out slope/intercept
[[slope]] = sess.run(A)  # 斜率
[[intercept]] = sess.run(b)  # 偏移

# Create fitted line
x = np.linspace(0, 3, num = 50)
ablineValues = []
for i in x:
    ablineValues.append(slope * i + intercept)
    pass

# Plot the fitted line over the data
setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms = 10, mew = 2, label = '山鸢尾')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label = '非山鸢尾')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle("图2-7：山鸢尾和非山鸢尾。\n实心直线是迭代1000次得到的线性分隔")
plt.xlabel('花瓣长度')
plt.ylabel("花瓣宽度")
plt.legend(loc = 'lower right')

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
