# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0309_logistic_regression.py
@Version    :   v0.1
@Time       :   2019-10-31 10:10
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0309，P62
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 Logistic 回归算法 解决二分类问题
"""
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

# The 'Low Birthrate Dataset' is a dataset provided by Univ. of Massachusetts at Amherst.
# It is a great dataset used for numerical prediction (birthweight)
# and logistic regression (binary classification, birthweight`<`2500g or not).
# Information about it is located here:",
# 这个地址的数据不允许访问
# birth_data_url = "https://www.umass.edu/statdata/statdata/data/lowbwt.txt"
# birth_file = requests.get(birth_data_url)
# birth_data = birth_file.text.split('\r\n')[5:]
with open("../Data/birthweight_data/birthweight.dat") as f:
    birth_file = f.read()
    pass
birth_data = birth_file.split('\n')
birth_header = [x for x in birth_data[0].split('\t') if len(x) >= 1]
# ToSee : 下面这个用法很有趣，值得关注
birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
x_vals = np.array([x[1:8] for x in birth_data])
y_vals = np.array([x[0] for x in birth_data])

# Split data into train/test = 80%/20%
data_count = len(x_vals)
train_indices = np.random.choice(data_count, int(round(data_count * 0.8)), replace = False)
test_indices = np.array(list(set(range(data_count)) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column ( min-max norm)
# 不能使用这个函数归一化训练数据和测试数据，因为必须使用归一化训练数据的标准去归一化测试数据，否则会造成数据变形
# 具体参考：《Python机器学习基础教程》，3.3.3. 对训练数据和测试数据进行相同的缩放
def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    return (m - col_min) / (col_max - col_min)


# 作者第二版已经做出了修改：08_logistic_regression.py
train_col_max = x_vals_train.max(axis = 0)
train_col_min = x_vals_train.min(axis = 0)
x_vals_train = np.nan_to_num((x_vals_train - train_col_min) / (train_col_max - train_col_min))
x_vals_test = np.nan_to_num((x_vals_test - train_col_min) / (train_col_max - train_col_min))

# Declare batch size
batch_size = 50
learning_rate = 0.01
iterations = 5001

# Initialize placeholders
x_data = tf.placeholder(shape = [None, 7], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape = [7, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = x_data @ A + b

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y_target))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Training loop
loss_vec, train_acc, test_acc = [], [], []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    temp_acc_train = sess.run(accuracy, feed_dict = {x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict = {x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)

    if (i + 1) % 250 == 0:
        print('-' * 50)
        print('Step #', i + 1, 'A = ', sess.run(A), 'b = ', sess.run(b))
        print('Loss = ', temp_loss)
        print('训练集精度 = ', temp_acc_train)
        print('测试集精度 = ', temp_acc_test)
        pass
    pass

# Plot loss over time
plt.figure()
plt.plot(loss_vec, 'b-')
plt.xlabel("迭代次数")
plt.ylabel("交叉熵损失函数")
plt.title("图3-11：Logistic 回归算法中 Sigmoid 损失函数\n学习率={}".format(learning_rate))

# Plot train and test accuracy
plt.figure()
plt.plot(train_acc, 'b-', label = "训练集")
plt.plot(test_acc, 'r--', label = "测试集")
plt.xlabel("迭代次数")
plt.ylabel("精度")
plt.ylim([-0.2, 1.2])
plt.legend(loc = 'lower right')
plt.title("图3-12：Logistic 回归算法中训练集和测试集的精确度")
# 不知道为什么，测试集的精确度那么高？
# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
