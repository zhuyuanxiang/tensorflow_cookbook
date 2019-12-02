# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0304_lin_reg_l1_vs_l2.py
@Version    :   v0.1
@Time       :   2019-10-30 16:04
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0304，P49
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现L1代价函数和L2代价函数的线性回归算法
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

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# 从这一节开始，都是使用花瓣宽度来拟合花萼长度
iris = load_iris()
x_vals = np.array([x[3] for x in iris.data])  # 花瓣宽度
y_vals = np.array([y[0] for y in iris.data])  # 花萼长度

# Declare batch size and number of iterations
batch_size = 25
learning_rate = 0.04  # Will not converge with learning rate at 0.4
iterations = 501

show_title("TensorFlow L1 Loss 代价损失函数的线性回归算法")

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare model operations
model_output = x_data @ A + b

# Declare loss functions
loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))

# Declare optimizers
my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l1 = my_opt_l1.minimize(loss_l1)

loss_vec_l1 = []
for i in range(iterations):
    # 随机选取数据
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    # 训练模型
    sess.run(train_step_l1, feed_dict = {x_data: rand_x, y_target: rand_y})
    # 计算损失
    temp_loss_l1 = sess.run(loss_l1, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec_l1.append(temp_loss_l1)
    if i % 50 == 0:
        print("Step #", i + 1, "A =", sess.run(A), "b=", sess.run(b))
        print("L1 Loss", temp_loss_l1)
        pass
    pass

[slope_l1] = sess.run(A)
[y_intercept_l1] = sess.run(b)
best_fit_l1 = []
for i in x_vals:
    best_fit_l1.append(slope_l1 * i + y_intercept_l1)
    pass

# -----------------------------------------------------------------
show_title("TensorFlow L2 Loss 代价损失函数的线性回归算法")

# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare model operations
model_output = x_data @ A + b

# Declare loss functions
loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))

# Declare optimizers
my_opt_l2 = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l2 = my_opt_l2.minimize(loss_l2)

loss_vec_l2 = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l2, feed_dict = {x_data: rand_x, y_target: rand_y})
    temp_loss_l2 = sess.run(loss_l2, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec_l2.append(temp_loss_l2)
    if i % 50 == 0:
        print("Step #", i + 1, "A =", sess.run(A), "b=", sess.run(b))
        print("L2 Loss", temp_loss_l2)
        pass
    pass

[slope_l2] = sess.run(A)
[y_intercept_l2] = sess.run(b)
best_fit_l2 = []
for i in x_vals:
    best_fit_l2.append(slope_l2 * i + y_intercept_l2)
    pass

plt.figure()
plt.plot(x_vals, y_vals, 'o', label = "数据点")
plt.plot(x_vals, best_fit_l1, 'b-', label = "L1损失函数的最佳匹配线")
plt.plot(x_vals, best_fit_l2, 'r-', label = "L2损失函数的最佳匹配线")
plt.xlabel('花瓣宽度')
plt.ylabel('花萼长度')
plt.title("图3-3：iris 数据集中的数据点 和 TensorFlow 拟合的直线")
plt.legend(loc = 'upper left')

# Plot loss over time
plt.figure()
plt.plot(loss_vec_l1, 'k-', label = 'L1 Loss')
plt.plot(loss_vec_l2, 'r--', label = 'L2 Loss')
plt.xlabel("迭代次数")
plt.ylabel("Loss 损失函数")
plt.legend(loc = 'upper right')
plt.title("图3-5：iris 数据线性回归的L1损失函数和L2损失函数\n学习率={}".format(learning_rate))

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
