# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0307_lasso_and_ridge_regression.py
@Version    :   v0.1
@Time       :   2019-10-30 18:15
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0307，P58
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 Lasso 回归 和 Ridge 回归 算法
"""
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

# -----------------------------------------------------------------
# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# 鸢尾花（3种），特征4种（花萼长度、花萼宽度、花瓣长度、花瓣宽度），150条数据
iris = load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare batch size
batch_size = 50
learning_rate = 0.001
iterations = 5001  # 多于2000次以后，好像才能收敛

# -----------------------------------------------------------------
show_title("TensorFlow Lasso Loss 代价损失函数的线性回归算法")

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output_lasso = x_data @ A + b
# model_output_lasso = tf.add(tf.matmul(x_data, A), b)

# Declare Lasso loss function(改良过的连续阶跃函数，Lasso回归的截止点设为0.9，即斜率系数不超过0.9)
# Lasso Loss = L2_Loss + heavy_side_step,
# Where heavy_side_step ~ 0 if A < constant, otherwise ~ 99
# 为了避免NaN的情况，要不保证A>0，要不调小正则参数，-100调为-10，99调为9.
# 但是对于这个实现的Lasso损失函数不太满意
lasso_param = tf.constant(0.9)
l2_loss = tf.reduce_mean(tf.square(y_target - model_output_lasso))
regular_param = 99. / (1. + tf.exp(-10. * (A - lasso_param)))
loss_lasso = l2_loss + regular_param
# heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lasso_param)))))
# regularization_param = tf.multiply(heavyside_step, 99.)
# loss_lasso = tf.add(l2, regularization_param)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt_lasso = tf.train.GradientDescentOptimizer(learning_rate)
train_step_lasso = my_opt_lasso.minimize(loss_lasso)

# Training loop
loss_vec_lasso = []
for i in range(iterations):
    # 通过分开打印发现，当A<0时，计算结果为NaN。
    # 而如果后面打印就会因为A已经被计算为NaN，而不知道NaN出现的原因了。
    if i % 100 == 0:
        print('-' * 50)
        print('Step #', i + 1, ' A = ', sess.run(A), ' b = ', sess.run(b), 'regular_param =', sess.run(regular_param))
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    feed_dict = {x_data: rand_x, y_target: rand_y}
    sess.run(train_step_lasso, feed_dict = feed_dict)
    temp_loss = sess.run(loss_lasso, feed_dict = feed_dict)
    loss_vec_lasso.append(temp_loss[0])
    if i % 100 == 0:
        print('Loss = ' + str(temp_loss))

# 取出最后一次计算的参数
[slope] = sess.run(A)
[y_intercept] = sess.run(b)
print("slope =", slope, "y_intercept =", y_intercept)

# 计算匹配的直线
best_fit_lasso = []
for i in x_vals:
    best_fit_lasso.append(slope * i + y_intercept)

# -----------------------------------------------------------------
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()
show_title("TensorFlow Ridge Loss 代价损失函数的线性回归算法")

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [1, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output_ridge = x_data @ A + b
# model_output_ridge = tf.add(tf.matmul(x_data, A), b)

# Declare the Ridge loss function
# Ridge loss = L2_loss + L2 norm of slope
# tf.reduce_mean()会把数据变成一维数据，所以需要把loss_ridge扩展成二维数据
ridge_param = tf.constant(1.)
l2_loss = tf.reduce_mean(tf.square(y_target - model_output_ridge))
loss_ridge = tf.expand_dims(l2_loss + ridge_param * tf.reduce_mean(tf.square(A)), 0)
# ridge_loss = tf.reduce_mean(tf.square(A))
# regularization_param = tf.multiply(ridge_param, ridge_loss)
# loss_ridge = tf.expand_dims(tf.add(l2, regularization_param), 0)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt_ridge = tf.train.GradientDescentOptimizer(learning_rate)
train_step_ridge = my_opt_ridge.minimize(loss_ridge)

# Training loop
loss_vec_ridge = []
for i in range(iterations):
    if (i + 1) % 100 == 0:
        print('-' * 50)
        print('Step #', i + 1, 'A = ', sess.run(A), 'b = ', sess.run(b))
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_ridge, feed_dict = {x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss_ridge, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec_ridge.append(temp_loss[0])
    if (i + 1) % 100 == 0:
        print('Loss = ' + str(temp_loss))

# 取出最后一次计算的参数
[slope] = sess.run(A)
[y_intercept] = sess.run(b)
print("slope =", slope, "y_intercept =", y_intercept)

# 计算匹配的直线
best_fit_ridge = []
for i in x_vals:
    best_fit_ridge.append(slope * i + y_intercept)

# -----------------------------------------------------------------
plt.figure()
plt.plot(x_vals, y_vals, 'o', label = "数据点")
plt.plot(x_vals, best_fit_lasso, 'b-', label = "Lasso 损失函数的最佳匹配线")
plt.plot(x_vals, best_fit_ridge, 'r-', label = "Ridge 损失函数的最佳匹配线")
plt.xlabel('花瓣宽度')
plt.ylabel('花萼长度')
plt.title("图3-3：iris 数据集中的数据点 和 TensorFlow 拟合的直线")
plt.legend(loc = 'upper left')

# -----------------------------------------------------------------
plt.figure()
plt.plot(loss_vec_lasso, 'b-', label = 'Lasso Loss')
plt.plot(loss_vec_ridge, 'r-', label = 'Ridge Loss')
plt.xlabel("迭代次数")
plt.ylabel("Loss 损失函数")
plt.legend(loc = 'upper right')
plt.title("图3-5：iris 数据线性回归的学习率={}\nLasso 损失函数和 Ridge 损失函数".format(learning_rate))

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
