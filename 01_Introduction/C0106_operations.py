# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0106_operations.py
@Version    :   v0.1
@Time       :   2019-10-29 14:11
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0106，P110
@Desc       :   TensorFlow 基础，声明操作
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

show_values(tf.div(3, 4), "tf.div(3,4) = 整数除")
show_values(tf.truediv(3, 4), "tf.truediv(3,4) = 浮点除")
show_values(tf.floordiv(3.0, 4.0), "tf.floordiv(3.0,4.0) = 浮点取整除")
show_values(tf.mod(22.0, 5.0), "tf.mod(22.0,5.0) = 取模")
# 张量点积--Compute the pairwise cross product
# 张量点积：即两个向量的叉乘，又叫向量积、外积、叉积，叉乘的运算结果是一个向量而不是一个标量。
# 两个向量的点积与这两个向量组成的坐标平面垂直。
show_values(tf.cross([1., 0., 0.], [0., 1., 0.]),
            "tf.cross([1., 0., 0.], [0., 1., 0.]) = 张量点积")
# 张量点积必须是三维的
# show_values(tf.cross([1., 0., 0., 0.], [0., 1., 0., 0.]),
#             "tf.cross([1., 0., 0.,0.], [0., 1., 0.,0.]) = 张量点积")

# ToSee：P11，数学函数列表

show_values(tf.div(tf.sin(3.1416 / 4.), tf.cos(3.1416 / 4.)),
            "tan(pi/4) = 1 = tf.div(tf.sin(3.1416/4.),tf.cos(3.1416/4.))")

test_nums = range(15)
# What should we get with list comprehension
expected_output = [3 * x * x - x + 10 for x in test_nums]
print('-' * 50)
print("[3 * x ^ 2 - x + 10 for x in test_nums] = ")
print(expected_output)


# 自定义函数
# 3x^2-x+10,x=11,=>
def custom_polynomial(value):
    # return tf.subtract(3 * tf.square(value), value) + 10
    return 3 * tf.square(value) - value + 10


show_values(custom_polynomial(11), "custom_polynomial(11) = 3x^2-x+10,x=11=>")
for num in test_nums:
    show_values(custom_polynomial(num), "custom_polynomial({})".format(num))

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
