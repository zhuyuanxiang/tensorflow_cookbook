# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0302_lin_reg_inverse.py
@Version    :   v0.1
@Time       :   2019-10-30 14:40
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0302，P45
@Desc       :   基于 TensorFlow 的线性回归，用 TensorFlow求伪逆矩阵
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

# 伪逆矩阵：X^+ = (A'*A)^(-1) * A' * b =  (t(A) * A)^(-1) * t(A) * b
#  t(A)：表示 A 的转置
# 利用数据直接求解的方法拟合模型，求解采用求逆矩阵的方法效率是比较低的，可以利用矩阵分解提高速度

# Create the data
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# Create design matrix
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# Create b matrix
b = np.transpose(np.matrix(y_vals))

# Create tensors
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# Matrix inverse solution
tA = tf.transpose(A_tensor)
tA_A = tf.matmul(tA, A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
solution = tf.matmul(tf.matmul(tA_A_inv, tA), b_tensor)
# 直接使用数学公式求解矩阵的逆
solution_eval = sess.run(solution)

# 提取系数
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print("slope:", slope)
print("y_intercept:", y_intercept)

# Get best fit line
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
    pass

# Plot the results
plt.plot(x_vals, y_vals, 'o', label = '数据')
plt.plot(x_vals, best_fit, 'r-', label = "最佳匹配线", linewidth = 3)
plt.legend(loc = 'upper left')
plt.suptitle("图3-1：通过矩阵求逆方法求解拟合直线和数据点")

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
