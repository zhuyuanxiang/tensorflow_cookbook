# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0402_linear_svm.py
@Version    :   v0.1
@Time       :   2019-10-31 15:23
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0402，P66
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 线性支持向量机 解决 软分类问题
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
# 鸢尾花（3种），特征4种（花萼长度、花萼宽度、花瓣长度、花瓣宽度），150条数据
iris = load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

# Split data into train/test sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 100
iterations = 5001

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [2, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = x_data @ A - b

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Declare loss function
# Loss = max(0, 1 - pred * actual ) + alpha * L2_norm( A )^2
# L2 regularization parameter, alpha
alpha = tf.constant([0.03])
# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., (1. - (model_output * y_target))))
regularization_term = alpha * l2_norm
# Put terms together
loss = classification_term + regularization_term

# Declare prediction function
prediction = tf.sign(model_output)  # tf.sign() 是指示函数
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec, train_accuracy, test_accuracy = [], [], []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict = {x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print("Step #", i + 1, "b =", sess.run(b), "A =", sess.run(A))
        print("Loss =", temp_loss)
        print("训练集精度 =", train_acc_temp)
        print("测试集精度 =", test_acc_temp)
        print("regularization_term =", sess.run(regularization_term))
        pass
    pass

# Extract coefficients
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2 / a1
y_intercept = b / a1

# Extract x1 and x2 vals
x1_vals = [d[1] for d in x_vals]

# Get best fit line
best_fit = []
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)
    pass

# Separate I. setosa
setaso_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setaso_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]
# 不能使用下面的方式简化，把数据写出来就明白为什么了。
# setaso_x, setaso_y = [[d[1], d[0]] for i, d in enumerate(x_vals) if y_vals[i] == 1]
# not_setosa_x, not_setosa_y = [[d[1], d[0]] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setaso_x, setaso_y, 'o', label = "I. setosa")
plt.plot(not_setosa_x, not_setosa_y, 'x', label = "Non-setosa")
plt.plot(x1_vals, best_fit, 'r-', label = "Linear Separator", linewidth = 3)
plt.ylim([0, 10])
plt.legend(loc = "lower right")
plt.xlabel("花瓣宽度")
plt.ylabel("花萼长度")

# Plot train/test accuracies
plt.figure()
plt.plot(train_accuracy, 'k-', label = '训练集精度')
plt.plot(test_accuracy, 'r--', label = '测试集精度')
plt.title('Train and Test Set Accuracies')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.legend(loc = 'lower right')

# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('迭代次数')
plt.ylabel('损失')
# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
