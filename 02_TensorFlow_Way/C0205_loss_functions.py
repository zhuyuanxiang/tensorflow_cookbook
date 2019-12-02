# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0205_loss_functions.py
@Version    :   v0.1
@Time       :   2019-10-29 14:59
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0205，P26
@Desc       :   TensorFlow 进阶，TensorFlow 实现损失函数
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

from tools import show_title, show_values

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


# 2.5 损失函数
def regression_loss_functions():
    show_title("回归算法的损失函数")

    ###### Numerical Predictions ######
    x_vals = tf.linspace(-1., 1., 500)
    target = tf.constant(0.)

    # L2 loss（平方损失函数）（欧拉损失函数）
    # L = (pred - actual)^2
    l2_y_vals = tf.square(target - x_vals)
    # show_values(l2_y_vals,"l2_y_vals")
    l2_y_out = sess.run(l2_y_vals)

    # L1 loss（绝对值损失函数）
    # L = abs(pred - actual)
    l1_y_vals = tf.abs(target - x_vals)
    # show_values(l1_y_vals,"l1_y_vals")
    l1_y_out = sess.run(l1_y_vals)

    # Pseudo-Huber loss
    # L = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
    delta1 = tf.constant(0.25)
    phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals) / delta1)) - 1.)
    phuber1_y_vals = tf.square(delta1) * (tf.sqrt(1. + tf.square((target - x_vals) / delta1)) - 1.)
    # show_values(phuber1_y_vals,"phuber1_y_vals")
    phuber1_y_out = sess.run(phuber1_y_vals)

    delta2 = tf.constant(5.)
    phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals) / delta2)) - 1.)
    phuber1_y_vals = tf.square(delta2) * (tf.sqrt(1. + tf.square((target - x_vals) / delta2)) - 1.)
    # show_values(phuber2_y_vals,"phuber2_y_vals")
    phuber2_y_out = sess.run(phuber2_y_vals)

    # Plot the output:
    # x_array = show_values(x_vals, "x_vals = ")
    x_array = sess.run(x_vals)
    plt.plot(x_array, l2_y_out, 'b-', label = 'L2 Loss')
    plt.plot(x_array, l1_y_out, 'r--', label = 'L1 Loss')
    plt.plot(x_array, phuber1_y_out, 'k-.', label = 'P-Huber Loss (0.25)')
    plt.plot(x_array, phuber2_y_out, 'g:', label = 'P-Huber Loss (5.0)')
    plt.ylim(-0.2, 0.4)
    plt.legend(loc = 'lower right', prop = {'size': 11})
    plt.title("图2-4：各种回归算法的损失函数")


def classfication_loss_functions():
    ###### Categorical Predictions ######
    x_vals = tf.linspace(-3., 5., 500)
    target = tf.constant(1.)
    targets = tf.fill([500, ], 1.)

    # Hinge loss
    # Use for predicting binary (-1, 1) classes
    # 主要用在评估支持向量机算法，也可以评估神经网络算法
    # 具体的公式需要根据算法中的情况决定，下面的公式仅供参考
    # L = max(0, 1 - (pred * actual))
    hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
    hinge_y_vals = tf.maximum(0., 1. - (target * x_vals))
    # hinge_y_out = show_values( hinge_y_vals,"hinge_y_vals")
    hinge_y_out = sess.run(hinge_y_vals)

    # Cross entropy loss
    # 交叉熵损失函数
    # L = -actual * (log(pred)) - (1-actual)(log(1-pred))
    xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
    xentropy_y_vals = -(target * tf.log(x_vals)) - ((1. - target) * tf.log(1. - x_vals))
    # xentropy_y_out = show_values( xentropy_y_vals,"xentropy_y_vals")
    xentropy_y_out = sess.run(xentropy_y_vals)

    # Sigmoid entropy loss
    # Sigmoid 交叉熵损失函数
    # L = -actual * (log(sigmoid(pred))) - (1-actual)(log(1-sigmoid(pred)))
    # or
    # L = max(actual, 0) - actual * pred + log(1 + exp(-abs(actual)))
    # 书中代码的 labels 和 logits 搞反了
    xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels = targets, logits = x_vals)
    # show_values(xentropy_sigmoid_y_vals,"xentropy_sigmoid_y_vals")
    xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

    # Weighted (Sigmoid) cross entropy loss
    # Sigmoid 加权交叉熵损失函数
    # L = targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
    # L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
    # or
    # L = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))
    weight = tf.constant(0.5)
    xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(
            targets = targets, logits = x_vals, pos_weight = weight)
    # show_values(xentropy_weighted_y_vals,"xentropy_weighted_y_vals")
    xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

    # Plot the output
    x_array = sess.run(x_vals)
    plt.plot(x_array, hinge_y_out, 'b-', label = 'Hinge损失函数')
    plt.plot(x_array, xentropy_y_out, 'r--', label = '两类交叉熵损失函数')
    plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label = 'Sigmoid 交叉熵损失函数')
    plt.plot(x_array, xentropy_weighted_y_out, 'g:', label = '加权交叉熵损失函数 (x0.5)')
    plt.ylim(-1.5, 3)
    # plt.xlim(-1, 3)
    plt.legend(loc = 'lower right', prop = {'size': 11})
    plt.title("图2-5：各种分类算法的损失函数")

    # Softmax entropy loss
    # Softmax 交叉熵损失函数
    # L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
    unscaled_logits = tf.constant([[1., -3., 10.]])
    target_dist = tf.constant([[0.1, 0.02, 0.88]])
    softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits = unscaled_logits, labels = target_dist)
    show_values(softmax_xentropy, "softmax_xentropy")
    # print(sess.run(softmax_xentropy))

    # Sparse entropy loss
    # 稀疏 Softmax 交叉熵损失函数
    # Use when classes and targets have to be mutually exclusive
    # L = sum( -actual * log(pred) )
    unscaled_logits = tf.constant([[1., -3., 10.]])
    sparse_target_dist = tf.constant([2])
    sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = unscaled_logits, labels = sparse_target_dist)
    show_values(sparse_xentropy, "sparse_xentropy")
    # print(sess.run(sparse_xentropy))


# -----------------------------------------------------------------
if __name__ == "__main__":
    # 2.5 损失函数
    # regression_loss_functions()
    classfication_loss_functions()

    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
