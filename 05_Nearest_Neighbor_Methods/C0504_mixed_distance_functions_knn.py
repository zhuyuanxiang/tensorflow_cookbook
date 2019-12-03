# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0504_mixed_distance_functions_knn.py
@Version    :   v0.1
@Time       :   2019-11-03 16:53
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0504，P98
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 混合距离计算
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

# Load the data
boston = load_boston()
# header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# used_h = ['CRIM',       'INDUS',         'NOX', 'RM', 'AGE', 'DIS',        'TAX', 'PTRATIO', 'B', 'LSTAT']
x_vals = np.delete(boston.data, [1, 3, 8, 13], axis = 1)
num_features = x_vals.shape[1]
y_vals = np.expand_dims(boston.target, axis = 1)

## Min-Max Scaling
x_vals = (x_vals - x_vals.min(axis = 0)) / x_vals.ptp(axis = 0)

## Create distance metric weight matrix weighted by standard deviation
# 4. 创建对角权重矩阵，矩阵提供了归一化的距离试题，其值为特征的标准差
weight_diagonal = x_vals.std(axis = 0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype = tf.float32)

# Split the data into train and test sets
train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * 0.8)), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size = len(x_vals_test)

# Placeholders
x_data_train = tf.placeholder(shape = [None, num_features], dtype = tf.float32)
y_target_train = tf.placeholder(shape = [None, 1], dtype = tf.float32)
x_data_test = tf.placeholder(shape = [None, num_features], dtype = tf.float32)
y_target_test = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Declare weighted distance metric
# 7. 声明距离函数
# Weighted - L2 = sqrt((x-y)^T * A * (x-y))
subtraction_term = x_data_train - tf.expand_dims(x_data_test, 1)
first_product = subtraction_term @ tf.tile(tf.expand_dims(weight_matrix, 0), [batch_size, 1, 1])
second_product = first_product @ tf.transpose(subtraction_term, perm = [0, 2, 1])
distance = tf.sqrt(tf.matrix_diag_part(second_product))

# Predict: Get min distance index (Nearest neighbor)
# 8. 计算每个测试数据点的距离，返回前K个最近领域（因为距离越近权重越高，故返回的是前K个负的距离值，即最小距离）
top_k_x_vals, top_k_indices = tf.nn.top_k(-distance, k = k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_x_vals, 1), 1)
x_sums_repeated = x_sums @ tf.ones([1, k], tf.float32)
x_vals_weights = tf.expand_dims(top_k_x_vals / x_sums_repeated, 1)
# tf.gather(params, indices, validate_indices=None, name=None, axis=0)将params中的数据按照indices中的定义进行汇总。
# tf.gather([0,1,2,3,4,5,6,7,8,9],[1,5,9]) --> [1,5,9]
# tf.gather([0,1,2,3],[0,0,2,1,3])-->[0, 0, 2, 1, 3]
top_k_y_vals = tf.gather(y_target_train, top_k_indices)
# tf.squeeze()：移除为1的维度，移除指定维为1的维度，如果指定维的维度不是1会报错。
prediction = tf.squeeze(x_vals_weights @ top_k_y_vals, axis = [1])

# Calculate MSE
# 9. 计算预测值的MSE，评估训练模型
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

plt.hist(predictions, bins, alpha = 0.5, label = '预测值')
plt.hist(y_batch, bins, alpha = 0.5, label = '实际值')
plt.title("图5-3：预测值和实际值对比的直方图（k-NN算法），k={}".format(k))
plt.xlabel("Med Home Value in $1,000s")
plt.ylabel('Frequency')
plt.legend(loc = "upper right")

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
