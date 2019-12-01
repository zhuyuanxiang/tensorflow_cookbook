# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0203_layering_nested_operations.py
@Version    :   v0.1
@Time       :   2019-10-29 14:28
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0203，P21
@Desc       :   TensorFlow 进阶， TensorFlow 的嵌入 Layer
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from tensorflow.python.framework import ops

from tools import show_values

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

# 2.3  层化的嵌入式操作
# -----------------------------------------------------------------
# 创建数据和占位符
my_value = np.array([1., 4., 8.])
my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
print("x_vals = ")
print(x_vals)
x_data = tf.placeholder(tf.float32)
show_values(x_data, '占位符', session = sess, feed_dict = {x_data: my_value})
show_values(x_data, '占位符', session = sess, feed_dict = {x_data: my_array})
x_data = tf.placeholder(tf.float32, shape = (3, 5))
show_values(x_data, '占位符', session = sess, feed_dict = {x_data: my_array})
x_data = tf.placeholder(tf.float32, shape = (3, None))
show_values(x_data, '占位符', session = sess, feed_dict = {x_data: my_array})

# -----------------------------------------------------------------
# 　创建常量矩阵
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

show_values(m1, "常量：m1", session = sess)
show_values(m2, "常量：m2", session = sess)
show_values(a1, "常量：a1", session = sess)

# 表示成计算图
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

prod1_symbol = x_data @ m1
add1_symbol = x_data @ m1 @ m2 + a1

# 使用计算图计算
for x_val in x_vals:
    show_values(prod1, "tf.matmul(x_data, m1)", feed_dict = {x_data: x_val}, session = sess)
    show_values(prod1_symbol, "x_data @ m1", feed_dict = {x_data: x_val}, session = sess)
    show_values(prod2, "tf.matmul(tf.matmul(x_data, m1), m2)", feed_dict = {x_data: x_val}, session = sess)
    show_values(add1, "tf.add(tf.matmul(tf.matmul(x_data, m1), m2), a1)", feed_dict = {x_data: x_val}, session = sess)
    show_values(add1_symbol, "x_data @ m1 @ m2 + a1", feed_dict = {x_data: x_val}, session = sess)

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
