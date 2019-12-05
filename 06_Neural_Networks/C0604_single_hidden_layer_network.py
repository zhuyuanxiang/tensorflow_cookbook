# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0604_single_hidden_layer_network.py
@Version    :   v0.1
@Time       :   2019-11-05 11:06
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0604，P117
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 单隐层神经网络
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
np.random.seed(42)
tf.set_random_seed(42)

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# numpy 1.16.4 is required
assert np.__version__ in ["1.16.5", "1.16.4"]
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()

# Implementing a one-layer Neural Network
# 神经网络参数配置不合适，很容易出现不收敛，或者过拟合。
iris = load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Normalize by a column (min-max norm)
# 基于列进行最大——最小归一化处理数据
# 不能使用这种方式归一化，必须统一使用训练集的标准对数据归一化，否则会造成数据变形
# def normalize_cols(m):
#     col_max = m.max(axis = 0)
#     col_min = m.min(axis = 0)
#     return (m - col_min) / (col_max - col_min)
#     # return (m - m.min(axis = 0)) / m.ptp(axis = 0)    # 等同于上面的函数
# x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
# x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

train_col_max = x_vals_train.max(axis = 0)
train_col_min = x_vals_train.min(axis = 0)
x_vals_train = np.nan_to_num((x_vals_train - train_col_min) / (train_col_max - train_col_min))
x_vals_test = np.nan_to_num((x_vals_test - train_col_min) / (train_col_max - train_col_min))

# Declare batch size
batch_size = 50  # 批处理数目过小，容易导致欠拟合，无法收敛
iteration = 1501

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 3], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for two Neural Network Layers
hidden_layer_nodes = 15
A1 = tf.Variable(tf.random_normal(shape = [3, hidden_layer_nodes]))  # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes]))  # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape = [1]))  # 1 bias for the output

# Declare model operations
# 使用 ReLU 很容易不收敛，改成leaky_relu 就会好很多，收敛速度快，而且比较稳定
hidden_output = tf.nn.leaky_relu(x_data @ A1 + b1)
final_output = tf.nn.leaky_relu(hidden_output @ A2 + b2)

# Declare loss function
loss = tf.reduce_mean(tf.square(y_target - final_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
train_loss_vec, test_loss_vec = [], []
for i in range(iteration):
    rand_indices = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_indices]
    rand_y = np.transpose([y_vals_train[rand_indices]])  # 加一维，然后转置
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    train_temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    # list的append()和extend()对于一维数据没有区别，
    # 对于多维数据，append()会将多维数据作为一个元素添加，extend()会将多维数据展开成相同维度与原始数据合并
    train_loss_vec.append(np.sqrt(train_temp_loss))
    # train_loss.extend(np.sqrt(train_temp_loss))

    test_temp_loss = sess.run(loss, feed_dict = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss_vec.append(np.sqrt(test_temp_loss))

    if i % 50 == 0:
        print('-' * 50)
        print("迭代次数 #", i + 1, ") 训练集损失函数值 = ", train_temp_loss)
        print("迭代次数 #", i + 1, ") 测试集损失函数值 = ", test_temp_loss)
        pass
    pass

# Plot loss (MSE) over time
plt.plot(train_loss_vec, 'b-', label = '训练集')
plt.plot(test_loss_vec, 'r--', label = '测试集')
plt.title('图6-4：训练集和测试集的损失函数（MSE）')
plt.xlabel("迭代次数")
plt.ylabel("损失函数值")
plt.legend(loc = 'upper right')

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
