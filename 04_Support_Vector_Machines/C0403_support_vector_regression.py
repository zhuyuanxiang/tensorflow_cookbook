# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0403_support_vector_regression.py
@Version    :   v0.1
@Time       :   2019-10-31 17:55
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0403，P72
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现支持向量机的线性回归
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

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)

# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
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

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Split data into train/test sets
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 50
learning_rate = 0.075
iterations = 5001

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
# model_output = tf.add(tf.matmul(x_data, A), b)
model_output = x_data @ A + b

# Declare loss function
# Loss = max(0, abs(target - predicted) + epsilon)
# 1/2 margin width parameter = epsilon
# 意味着如果一个数据点在epsilon定义的区间内，则损失等于0
epsilon = tf.constant([0.5])
# Margin ter in loss
# loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))
loss = tf.reduce_mean(tf.maximum(0., (tf.abs(y_target - model_output) - epsilon)))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
train_loss = []
test_loss = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_train_loss = sess.run(loss, feed_dict = {
            x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])
    })
    train_loss.append(temp_train_loss)

    temp_test_loss = sess.run(loss, feed_dict = {
            x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])
    })
    test_loss.append(temp_test_loss)
    if (i + 1) % 100 == 0:
        print('-----------')
        print('Generation: ' + str(i + 1))
        print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Train Loss = ' + str(temp_train_loss))
        print('Test Loss = ' + str(temp_test_loss))
        pass
    pass

# Extract Coefficients
[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

# Get best fit line
best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
    best_fit_upper.append(slope * i + y_intercept + width)
    best_fit_lower.append(slope * i + y_intercept - width)
    pass
best_fit_upper = np.add(best_fit, width)
best_fit_lower = np.subtract(best_fit, width)

plt.figure()
plt.plot(x_vals, y_vals, 'o', label = "数据点")
plt.plot(x_vals, best_fit, 'r-', label = "SVM Regression Line", linewidth = 3)
plt.plot(x_vals, best_fit_upper, 'b--', linewidth = 2)
plt.plot(x_vals, best_fit_lower, 'b--', linewidth = 2)
plt.ylim([0, 10])
plt.legend(loc = "lower right")
plt.title("图4-5：基于 iris 数据集的支持向量机回归")
plt.xlabel("花瓣宽度")
plt.ylabel("花萼长度")

plt.figure()
plt.plot(train_loss, 'b-', label = "训练集损失值")
plt.plot(test_loss, 'r--', label = "测试集损失值")
plt.title("图4-6：每次迭代的支持向量机回归的损失值")
plt.xlabel("迭代次数")
plt.ylabel("L2 损失值")
plt.legend(loc = "upper right")
# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
