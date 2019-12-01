# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0104_placeholders.py
@Version    :   v0.1
@Time       :   2019-10-29 11:41
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0104，P6
@Desc       :   TensorFlow 基础，使用占位符和变量
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
from tensorflow.python.framework import ops

# 设置数据显示的精确度为小数点后3位
from tools import show_title, show_values

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


# 变量必须在 session 中初始化，才可以使用
def declare_variable():
    show_title("TensorFlow 使用变量")

    # Declare a variable
    my_var = tf.Variable(tf.zeros([1, 20]))

    print("全局初始化变量")
    initialize_op = tf.global_variables_initializer()  # Initialize operation
    sess.run(initialize_op)  # Run initialization of variable

    print("my_var = ", my_var)
    print("sess.run(my_var)", sess.run(my_var))

    show_values(initialize_op, "initialize_op", session = sess)
    # 不同的 session ，不同的环境初始化。
    show_values(my_var, "my_var", session = sess)

    print('-' * 50)
    print("每个变量独自初始化。。。")
    first_var = tf.Variable(tf.zeros([2, 3]))
    print("first_var", first_var)
    print("first_var.initializer = ", first_var.initializer)
    print("sess.run(first_var.initializer) = ", sess.run(first_var.initializer))
    print("sess.run(first_var) = \n", sess.run(first_var))
    show_values(first_var, "first_var", session = sess)
    show_values(first_var.initializer, "first_var.initializer", session = sess)

    print('-' * 50)
    second_var = tf.Variable(tf.ones_like(first_var))
    print("second_var", second_var)
    print("second_var.initializer", second_var.initializer)
    print("sess.run(second_var.initializer) = ", sess.run(second_var.initializer))
    print("sess.run(second_var) = \n", sess.run(second_var))
    show_values(second_var.initializer, "second_var.initializer", session = sess)
    show_values(second_var, "second_var", session = sess)


def declare_placeholder():
    show_title("TensorFlow 使用占位符")

    x = tf.placeholder(tf.float32, shape = (4, 4))
    y = tf.identity(x)  # 返回占位符传入的数据本身
    z = tf.matmul(y, x)

    x_vals = np.random.rand(4, 4)
    print("随机生成的原始张量（x_vals） = ")
    print(x_vals)
    show_values(y, "tf.identity(tf.placeholder(tf.float32, shape = (4,4)))", session = sess,
                feed_dict = {x: x_vals})
    show_values(tf.matmul(x_vals, x_vals), "tf.matmul(x_vals,x_vals)", session = sess)
    show_values(z, "tf.matmul(y,tf.placeholder(tf.float32, shape = (4,4)))", session = sess,
                feed_dict = {x: x_vals})

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph_def)


if __name__ == "__main__":
    # declare_variable()

    declare_placeholder()
    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
