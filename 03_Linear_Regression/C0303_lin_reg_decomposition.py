# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0303_lin_reg_decomposition.py
@Version    :   v0.1
@Time       :   2019-10-30 15:09
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0303，P47
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现矩阵分解（Cholesky分解）提高求逆的速度
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

# Given Ax=b, and a Cholesky decomposition such that
#  A = L*L' then we can get solve for x via
# 1) L*y=t(A)*b --> 求解y
# 2) L'*x=y --> 求解x
# 因为A不是对称半正定矩阵，所以需要使用t(A)*A将之转成对称半正定矩阵
# 使用 Cholesky 分解可以提高矩阵求逆的速度
# 重点与难点：Cholesky 分解的数学理解
# Create the data
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# Create design matrix
# x_vals_column = np.transpose(np.matrix(x_vals))
# ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
x_vals_column = np.transpose([x_vals])
ones_column = np.transpose([np.repeat(1, 100)])
A = np.column_stack((x_vals_column, ones_column))

# Create b matrix
# b = np.transpose(np.matrix(y_vals))
b = np.transpose([y_vals])

# Create tensors
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# Find Cholesky Decomposition
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)

# Solve L*y=t(A)*b
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)

# Solve L' * y = sol1
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

solution_eval = sess.run(sol2)

# Extract coefficients
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

# Get best fit line
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)

# Plot the results
plt.plot(x_vals, y_vals, 'o', label = '数据')
plt.plot(x_vals, best_fit, 'r-', label = '最佳匹配线', linewidth = 3)
plt.legend(loc = 'upper left')
plt.suptitle("图3-2：通过矩阵分解求解拟合直线和数据点")
# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
