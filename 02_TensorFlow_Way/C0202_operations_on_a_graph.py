# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0202_operations_on_a_graph.py
@Version    :   v0.1
@Time       :   2019-10-29 14:21
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0202，P20
@Desc       :   TensorFlow 进阶，计算图中的操作
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

# 2.2 计算图
# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
print("x_vals = ", x_vals)
# placeholder() 占位符
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print('=' * 50)
    print("x_val = ", x_val)
    show_values(my_product,
                "tf.multiply(tf.placeholder(tf.float32), tf.constant(3.)) = ",
                feed_dict = {x_data: x_val})
    pass

print('=' * 50)
my_product = x_data * m_const
replace_dict = {x_data: 15.}
print("replace_dict = ", replace_dict)
with sess.as_default():
    print("my_product.eval(feed_dict = replace_dict)", my_product.eval(feed_dict = replace_dict))
    pass
print("sess.run(my_product,feed_dict = replace_dict)", sess.run(my_product, feed_dict = replace_dict))

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
