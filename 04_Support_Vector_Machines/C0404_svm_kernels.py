# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0404_svm_kernels.py
@Version    :   v0.1
@Time       :   2019-11-01 16:38
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0404，P77
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 基于核函数的支持向量机
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

# 从这一节开始，主要难点在于高斯核函数推导过程的实现。
# 书中没有详细介绍核函数算法的公式
# 书中的公式与代码中的实现无法匹配
# 虽然代码正确运行，但是没有推导无法真正理解原理，也就很正确推广
# 4.5 对非线性核的实现也存在同样的问题
# 4.6 对多类别分类的实现也存在同样的问题

# Generate non-linear data
# 数据量和批量数据个数会影响最终判断的效果
# 如果数据量与批量数据个数相等，那么不在聚焦区内的数据默认使用外面那个圈的值（-1）
# 如果数据量大于批量数据个数，那么不在聚焦区内的数据默认使用里面那个圈的值（1）
(x_vals, y_vals) = sklearn.datasets.make_circles(n_samples = 350, factor = .5, noise = .1)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

# Declare batch size
batch_size = 50
learning_rate = 0.002
iterations = 1500

# Initialize placeholders
# [None,2]表示需要提供n个2维数据，即n*2维数据
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
prediction_grid = tf.placeholder(shape = [None, 2], dtype = tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape = [1, batch_size]))

# Apply kernel
# 线性核没有高斯核的精度好。

# Gaussian (RBF) kernel
# ToDo: 没有具体的公式，不明白计算的原理。
gamma = tf.constant(-50.0)
# dist是列向量，先将x_data每个元素求平方，然后把所有结果相加，再全部转换为n个一维数据
dist = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
data_dist = 2. * (x_data @ tf.transpose(x_data))  # data_dist是矩阵，一个批次的训练数据*一个批次的训练数据的转置，
sq_dists = dist - data_dist + tf.transpose(dist)  # sq_dists被处理成矩阵对角线全部为0
my_kernel = tf.exp(gamma * tf.abs(sq_dists))  # exp{-gamma*||x-y||^2}

# Linear Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# Compute SVM model
model_output = b @ my_kernel
first_term = tf.reduce_sum(b)
b_vec_cross = tf.transpose(b) @ b
y_target_cross = y_target @ tf.transpose(y_target)
second_term = tf.reduce_sum((my_kernel * (b_vec_cross * y_target_cross)))
loss = tf.negative(first_term - second_term)

# Create Prediction Kernel
# Gaussian(RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_dist = tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))
pred_sq_dist = tf.add(tf.subtract(rA, pred_dist), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# Linear Prediction Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

prediction_output = (tf.transpose(y_target) * b) @ pred_kernel
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
rand_x = []
rand_y = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc = sess.run(accuracy, feed_dict = {
            x_data: rand_x, y_target: rand_y, prediction_grid: rand_x
    })
    batch_accuracy.append(temp_acc)

    if (i + 1) % 250 == 0:
        print("Step #", i + 1)
        print("Loss = ", temp_loss)
        print("Accuracy = ", temp_acc)
        pass
    pass

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
# Translates slice objects to concatenation along the second axis.
# 将所有数据都转换到矩阵的第二维中。数值默认为一个2维的矩阵数据
# np.c_[np.array([[[1,2,3]]]), 0, 0, np.array([[4,5,6]])] --> [[1,2,3,0,0,4,5,6]
# 将所有同维的矩阵合并到同维的一个矩阵中。
# np.c_[np.array([[[1,2,3]]]), np.array([[[4,5,6]]])] --> [[[1,2,3,4,5,6]]]
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = sess.run(
        prediction, feed_dict = {
                x_data: rand_x, y_target: rand_y, prediction_grid: grid_points
        }).reshape(xx.shape)

# Plot points and grid
plt.figure()
plt.contourf(xx, yy, grid_predictions, cmap = plt.cm.Paired, alpha = 0.8)
plt.plot(class1_x, class1_y, 'ro', label = "Class 1")
plt.plot(class2_x, class2_y, 'bx', label = "Class -1")
plt.title("图4-8：使用非线性的高斯核函数 SVM 在非线性可分的数据集上进行分割")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = "lower right")
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# Plot batch accuracy
plt.figure()
plt.plot(batch_accuracy, 'b-', label = 'Accuracy')
plt.title("批处理的精确度")
plt.xlabel("迭代次数")
plt.ylabel("精确度")
plt.legend(loc = "lower right")

# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'b-')
plt.title("每次迭代的损失代价")
plt.xlabel("迭代次数")
plt.ylabel("损失代价")

# Evaluate on new/unseen data points
# New data points:
new_points = np.array(
        [[0.1751517, - 0.2138658], [0.1951517, - 0.2138658], [0.2151517, - 0.2138658], [0.2351517, - 0.2138658],
         [0.2551517, - 0.2138658], [0.2751517, - 0.2138658], [0.2951517, - 0.2138658], [0.3151517, - 0.2138658],
         [0.3351517, - 0.2138658], [0.3551517, - 0.2138658], [0.3751517, - 0.2138658], [0.3951517, - 0.2138658],
         [0.4151517, - 0.2138658], [0.4351517, - 0.2138658], [0.4551517, - 0.2138658], ])
# new_points = np.array([(-0.75, -0.75), (-0.5, -0.5), (-0.25, -0.25), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)])
[evaluations] = sess.run(prediction, feed_dict = {
        x_data: rand_x, y_target: rand_y, prediction_grid: new_points
})

for ix, p in enumerate(new_points):
    print('{} : class = {}'.format(p, evaluations[ix]))
pass

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
