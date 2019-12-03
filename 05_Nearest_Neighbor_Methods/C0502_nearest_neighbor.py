# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0502_nearest_neighbor.py
@Version    :   v0.1
@Time       :   2019-11-02 15:43
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0502，P90
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 最近邻算法
"""
# common imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from sklearn.datasets import load_boston
from tensorflow.python.framework import ops

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)
# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)

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

# k-Nearest Neighbor
#
# Data:
# ----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
# ------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

# Load the data
boston = load_boston()
# header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# used_h = ['CRIM',       'INDUS',         'NOX', 'RM', 'AGE', 'DIS',        'TAX', 'PTRATIO', 'B', 'LSTAT']
x_vals = np.delete(boston.data, [1, 3, 8, 13], axis = 1)
num_features = x_vals.shape[1]
y_vals = np.expand_dims(boston.target, axis = 1)

# Min-Max Scaling
# 先变换尺度，再分割数据集的方式是错误的，
# 因为数据没有严重变形，因此不会给结果带来太大问题，故不做修改。
# 利用测试集修正模型参数导致的误差称作将测试集的信息“泄漏”到模型中。
# 具体参考：《Python机器学习基础教程》，5.2.2 参数过拟合的风险与验证集
# numpy.ptp()函数：返回沿轴的值的范围(最大值 - 最小值)。
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# 4. Split the data into train and test sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 5. Declare k-value and batch size
k = 4
batch_size = len(x_vals_test)

# 6. Placeholders
x_data_train = tf.placeholder(shape = [None, num_features], dtype = tf.float32)
x_data_test = tf.placeholder(shape = [None, num_features], dtype = tf.float32)
y_target_train = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y_target_test = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# 7. Declare distance metric.
# L1 norm
distance = tf.reduce_sum(tf.abs(x_data_train - tf.expand_dims(x_data_test, 1)), reduction_indices = 2)
# L2 norm
# distance = tf.sqrt(tf.reduce_sum(tf.square(x_data_train - tf.expand_dims(x_data_test, 1)), reduction_indices = 2))

# 8. Predict: Get min distance index (Nearest neighbor)
# 最近邻算法是聚类算法（无监督的），不需要损失函数最小化
# 使用了加权距离：权重=距离的归一化倒数
# top_k()寻找k个最近的数据，负数即距离越小，值越大
top_k_x_vals, top_k_indices = tf.nn.top_k(-distance, k = k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_x_vals, 1), 1)
x_sums_repeated = x_sums @ tf.ones([1, k], tf.float32)
x_vals_weights = tf.expand_dims(top_k_x_vals / x_sums_repeated, 1)
# tf.gather(params, indices, validate_indices=None, name=None, axis=0)将params中的数据按照indices中的定义进行汇总。
# tf.gather([0,1,2,3,4,5,6,7,8,9],[1,5,9]) --> [1,5,9]
# tf.gather([9,8,7,6,5,4,3,2,1,0],[1,5,9]) --> [8,4,0]
top_k_y_vals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(x_vals_weights @ top_k_y_vals, axis = [1])

# Calculate MSE
mse = tf.reduce_sum(tf.square(prediction - y_target_test)) / batch_size
num_loops = int(np.ceil(len(x_vals_test) / batch_size))
predictions, x_batch, y_batch, feed_dict = [], [], [], {}
for i in range(num_loops):
    min_index = i * batch_size
    max_index = min((i + 1) * batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    feed_dict = {
            x_data_train: x_vals_train,
            x_data_test: x_batch,
            y_target_train: y_vals_train,
            y_target_test: y_batch
    }
    predictions = sess.run(prediction, feed_dict = feed_dict)
    batch_mse = sess.run(mse, feed_dict = feed_dict)
    print("Batch #", i + 1)
    print("MSE:", np.round(batch_mse, 3))
    pass

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha = 0.5, label = '预测')
plt.hist(y_batch, bins, alpha = 0.5, label = '真实')
plt.title("图5-1：预测值和真实值对比的直方图（k-NN算法），k={}".format(k))
plt.xlabel("Med Home Value in $1,000s")
plt.ylabel('频率')
plt.legend(loc = "upper right")

# show_values(x_sums, "x_sums", feed_dict = feed_dict, session = sess)
# show_values(x_sums_repeated, "x_sums_repeated", feed_dict = feed_dict, session = sess)
# show_values(y_target_train, "y_target_train", feed_dict = feed_dict, session = sess)
# show_values(top_k_indices, "top_k_indices", feed_dict = feed_dict, session = sess)
# show_values(top_k_y_vals, "top_k_y_vals", feed_dict = feed_dict, session = sess)


# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
