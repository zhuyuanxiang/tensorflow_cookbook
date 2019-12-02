# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0103_declaring_tensors.py
@Version    :   v0.1
@Time       :   2019-10-29 11:18
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0103，P3
@Desc       :   TensorFlow 基础，张量
"""
# common imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
from tensorflow.python.framework import ops

from tools import show_title, show_values

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)
# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 初始化默认的计算图
ops.reset_default_graph()
# Open graph session
sess = tf.Session()


# -----------------------------------------------------------------------
# 1.3 声明张量
def declare_fix_tensor():
    print("1. 固定张量")
    row_dim, col_dim = (3, 2)
    print("张量行={}，列={}".format(row_dim, col_dim))

    zeros_tsr = tf.zeros([row_dim, col_dim])
    with sess.as_default():
        print(zeros_tsr.eval())
    show_values(zeros_tsr, 'zeros_tsr', session = sess)
    print("\t创建指定维度的零张量")

    ones_tsr = tf.ones([row_dim, col_dim])
    show_values(ones_tsr, "ones_tsr", session = sess)
    print("\t创建指定维度的单位张量")

    filled_tsr = tf.fill([row_dim, col_dim], 42)
    show_values(filled_tsr, "filled_tsr", session = sess)
    print("\t创建指定维度的常数填充的张量")

    print('=' * 50)
    print("\t创建常数张量")
    const_tsr = tf.constant([8, 6, 7, 5, 3, 0, 9])
    show_values(const_tsr, "const_tsr", session = sess)
    print("\t\t创建一维常量")

    constant_tsr = tf.constant([[1, 2, 3], [4, 5, 6]])
    show_values(constant_tsr, "constant_tsr", session = sess)
    print("\t\t创建二维常量")

    const_fill_tsr = tf.constant(-1, shape = [row_dim, col_dim])
    show_values(const_fill_tsr, "const_fill_tsr", session = sess)
    print("\t\t填充二维常量")
    pass


# -----------------------------------------------------------------------
# 2. 相似形状的张量
# shaped like other variable
def declare_similar_tensor():
    show_title("TensorFlow 声明相似形状的张量")

    int_const_tsr = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    show_values(int_const_tsr, "整数常数张量")

    zeros_similar_tsr = tf.zeros_like(int_const_tsr)
    show_values(zeros_similar_tsr, "相似形状的零张量")

    ones_similar_tsr = tf.ones_like(int_const_tsr)
    show_values(ones_similar_tsr, "相似形状的单位张量", session = sess)

    print('=' * 50)
    print("运算符重载")
    add_tsr = int_const_tsr + int_const_tsr
    show_values(add_tsr, "两个张量相加(int_const_tsr + int_const_tsr)", session = sess)

    multiply_tsr = int_const_tsr * int_const_tsr
    show_values(multiply_tsr, "两个张量相乘(int_const_tsr * int_const_tsr)", session = sess)

    neg_constant_tsr = -int_const_tsr
    show_values(neg_constant_tsr, "负张量(-int_const_tsr)", session = sess)

    number_multiply_tsr = 2 * int_const_tsr
    show_values(number_multiply_tsr, "数乘以张量(2 * int_const_tsr)", session = sess)

    abs_tsr = abs(neg_constant_tsr)
    show_values(abs_tsr, "张量取整(abs(neg_constant_tsr))", session = sess)

    minus_tsr = abs_tsr - neg_constant_tsr
    show_values(minus_tsr, "两个张量相减(abs_tsr - neg_constant_tsr)", session = sess)

    divide_tsr = multiply_tsr / neg_constant_tsr
    show_values(divide_tsr, "两个张量相除divide_tsr =(multiply_tsr / neg_constant_tsr)", session = sess)

    print('=' * 50)
    real_const_tsr = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype = tf.float64)
    show_values(real_const_tsr, "浮点常数张量(real_const_tsr)", session = sess)

    a_one = real_const_tsr * real_const_tsr  # 这个是矩阵数乘，不是矩阵乘法
    show_values(a_one, "两个张量矩阵数乘(real_const_tsr*real_const_tsr)", session = sess)

    a_floor_div = real_const_tsr // divide_tsr
    show_values(a_floor_div, "两个张量整除(real_const_tsr // divide_tsr)", session = sess)

    a_mod = real_const_tsr % divide_tsr
    show_values(a_mod, "两个张量取余(real_const_tsr % divide_tsr)", session = sess)

    a_power = real_const_tsr ** real_const_tsr
    show_values(a_power, "两个张量取幂(real_const_tsr ** divide_tsr)", session = sess)

    a_matrix_multipy = real_const_tsr @ real_const_tsr
    show_values(a_matrix_multipy, "两个张量矩阵乘(real_const_tsr @ real_const_tsr)")


# 3. 序列张量
def declare_seq_tensor():
    show_title("TensorFlow 声明序列张量")

    linear_seq_tsr = tf.linspace(start = 0.0, stop = 1.0, num = 3)
    show_values(linear_seq_tsr, "浮点序列张量", session = sess)

    integer_seq_tsr = tf.range(start = 6, limit = 15, delta = 3)
    show_values(integer_seq_tsr, "整数序列张量", session = sess)


# 4. 随机张量
def declare_random_tensor():
    show_title("TensorFlow 声明随机张量")

    row_dim, col_dim = (6, 5)

    randunif_tsr = tf.random_uniform([row_dim, col_dim], minval = 0, maxval = 1)
    show_values(randunif_tsr, "均匀分布的随机数", session = sess)

    randnorm_tsr = tf.random_normal([row_dim, col_dim], mean = 0.0, stddev = 1.0)
    show_values(randnorm_tsr, "正态分布的随机数", session = sess)

    runcnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean = 0.0, stddev = 1.0)
    show_values(runcnorm_tsr, "带有指定边界的正态分布的随机数", session = sess)

    shuffled_output = tf.random_shuffle(randunif_tsr)
    show_values(shuffled_output, "张量随机排序", session = sess)

    cropped_output = tf.random_crop(randunif_tsr, [3, 4])
    show_values(cropped_output, "张量的随机剪裁", session = sess)

    # 这个是剪裁图片的例子，没有图片，不能执行
    # cropped_image = tf.random_crop(my_image, [height / 2, width / 2, 3])
    # my_var = tf.Variable(tf.zeros([row_dim, col_dim]))


if __name__ == "__main__":
    # declare_fix_tensor()
    #
    # declare_similar_tensor()

    # declare_seq_tensor()

    # declare_random_tensor()
    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
