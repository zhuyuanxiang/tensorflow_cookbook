# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0405_nonlinear_svm.py
@Version    :   v0.1
@Time       :   2019-11-02 11:07
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0405，P82
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 非线性支持向量机（基于高斯核）
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

# 与上一节一样，同样是使用高斯核函数实现非线性支持向量机，区别在于对真实的数据进行分类
# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

# Declare batch size
batch_size = 150

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
prediction_grid = tf.placeholder(shape = [None, 2], dtype = tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape = [1, batch_size]))

# Gaussian (RBF) kernel
# Gamma值越大，每个数据点对分类边界的影响就越大
gamma = tf.constant(-25.)
dist = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
sq_dists = dist - (2. * (x_data @ tf.transpose(x_data))) + tf.transpose(dist)
my_kernel = tf.exp(gamma * tf.abs(sq_dists))

# Compute SVM Model
model_output = b @ my_kernel
first_term = tf.reduce_sum(b)
b_vec_cross = tf.transpose(b) @ b
y_target_cross = y_target @ tf.transpose(y_target)
second_term = tf.reduce_sum(my_kernel * b_vec_cross * y_target_cross)
loss = second_term - first_term

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = rA - (2. * (x_data @ tf.transpose(prediction_grid))) + tf.transpose(rB)
pred_kernel = tf.exp(gamma * tf.abs(pred_sq_dist))

prediction_output = (tf.transpose(y_target) * b) @ pred_kernel
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
rand_x = []
rand_y = []

for i in range(300):
    rand_index = np.random.choice(len(x_vals), size = batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc = sess.run(accuracy, feed_dict = {
            x_data: rand_x,
            y_target: rand_y,
            prediction_grid: rand_x
    })
    batch_accuracy.append(temp_acc)

    if (i + 1) % 25 == 0:
        print("Step #", i + 1)
        print("Loss = ", temp_loss)
        print("Accuracy = ", temp_acc)
        pass
    pass

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_predictions] = sess.run(prediction, feed_dict = {
        x_data: rand_x,
        y_target: rand_y,
        prediction_grid: grid_points
})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.figure()
plt.contourf(xx, yy, grid_predictions, cmap = plt.cm.Paired, alpha = 0.8)
plt.plot(class1_x, class1_y, 'ro', label = 'I. setosa')
plt.plot(class2_x, class2_y, 'bx', label = 'Non setosa')
plt.title("图4-9：采用高斯核函数的 SVM 的 山鸢尾花（I. setosa）的分类器效果图\n"
          "Gamma = {}".format(sess.run(gamma)))
plt.xlabel('花瓣长度')
plt.ylabel('花萼宽度')
plt.legend(loc = 'lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])

# # Plot batch accuracy
# plt.figure()
# plt.plot(batch_accuracy, 'b-', label = 'Accuracy')
# plt.title("批处理的精确度")
# plt.xlabel("迭代次数")
# plt.ylabel("精确度")
# plt.legend(loc = "lower right")
#
# # Plot loss over time
# plt.figure()
# plt.plot(loss_vec, 'b-')
# plt.title("每次迭代的损失代价")
# plt.xlabel("迭代次数")
# plt.ylabel("损失代价")


# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
