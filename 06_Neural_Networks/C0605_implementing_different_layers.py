# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0605_implementing_different_layers.py
@Version    :   v0.1
@Time       :   2019-11-06 18:03
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec06，P1
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 
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

from tools import show_title

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)
# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)
tf.set_random_seed(42)

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


# Implementing Different Layers
# ---------------------------------------
# 使用 TensorFlow 实现以下层：
#
# The layers of interest are:
#  (1) 卷积层：Convolutional Layer
#  (2) 激活层：Activation Layer
#  (3) 最大池化层：Max-Pool Layer
#  (4) 全连接层：Fully Connected Layer
#
# 下面生成两种类型的数据
#  1. 1-D data set (row of data)序列数据
#  2. 2-D data set (similar to picture)图片数据
# 具体参考《深度学习》第9章
# ---------------------------------------------------|
# -------------------1D-data-------------------------|
# ---------------------------------------------------|
def convolution_1d_data():
    show_title("1-D 序列数据")

    # Reset Graph
    ops.reset_default_graph()
    sess = tf.Session()

    # Generate 1D data
    data_size = 25
    data_1d = np.random.normal(size = data_size)

    # Placeholder
    x_input_1d = tf.placeholder(dtype = tf.float32, shape = [data_size])

    # --------Convolution--------
    def conv_layer_1d(input_1d, my_filter):
        # Tensorflow 的层函数是为四维数据设计的（batch size, width, height, channels）
        # expand_dims() 用于扩展维度；squeeze() 用于降低维度
        input_2d = tf.expand_dims(input_1d, 0)
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)

        # tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        # input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
        # filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
        # strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
        # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
        # use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
        convolution_output = tf.nn.conv2d(input_4d, filter = my_filter, strides = [1, 1, 1, 1], padding = 'VALID')
        conv_output_1d = tf.squeeze(convolution_output)  # 去除增加的维度
        return conv_output_1d

    # 创建卷积使用的滤波器，自动生成的参数，训练中会改变
    my_filter = tf.Variable(tf.random_normal(shape = [1, 5, 1, 1]))
    my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

    # --------Activation--------
    def activation(input_1d):
        return tf.nn.relu(input_1d)

    my_activation_output = activation(my_convolution_output)

    # --------Max Pool--------
    def max_pool(input_1d, width):
        # Tensorflow 的 max_pool() 函数使用 4D 数据：[batch_size, width, height, channels]
        # batch：为数据的批量个数
        # in_height：图片高度
        # in_weight 为图片宽度
        # in_channel 为图片的通道数，灰度图该值为1（0到255表示从黑到白），彩色图为3（红、绿、蓝）
        # 因此需要把数据从2D扩展到4D
        input_2d = tf.expand_dims(input_1d, 0)
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)

        # tf.nn.max_pool(value, ksize, strides, padding, name=None)
        # value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
        # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        # padding：和卷积类似，可以取'VALID' 或者'SAME'
        # 返回：Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
        pool_output = tf.nn.max_pool(
                input_4d, ksize = [1, 1, width, 1], strides = [1, 1, 1, 1], padding = 'VALID')
        pool_output_1d = tf.squeeze(pool_output)
        return pool_output_1d

    my_max_pool_output = max_pool(my_activation_output, width = 5)

    # --------Fully Connected--------
    def fully_connected(input_layer, num_outputs):
        # 全连接层需要把2D数据展开成1D数据
        # TF1.2.1 tf.stack() 替代 tf.pack()
        weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))  # 权重矩阵的形状
        weight = tf.random_normal(weight_shape, stddev = 0.1)  # 初始化权重矩阵
        bias = tf.random_normal(shape = [num_outputs])  # 初始化偏移量
        input_layer_2d = tf.expand_dims(input_layer, 0)  # 把 flat_input 展开成2D，才能与 weight 矩阵相乘
        full_output = (input_layer_2d @ weight) + bias
        # full_output_2d = tf.squeeze(full_output)  # 输出是1D，不需要再压缩维度
        full_output_1d = full_output
        return full_output_1d

    my_full_output = fully_connected(my_max_pool_output, 5)

    # Run graph
    # Initialize Variables
    init = tf.global_variables_initializer()
    sess.run(init)

    feed_dict = {x_input_1d: data_1d}

    # Convolution Output
    print("Data = 序列长度 25")
    print(data_1d)
    print('-' * 50)
    print("\nFilter（滤波器） = ", my_filter.get_shape())
    print(sess.run(my_filter))

    print('-' * 50)
    print('\nInput = 序列长度 25')
    print("卷积输出（序列长度=21，滤波器长度=5，步长=1）：")
    run_my_convolution_output = sess.run(my_convolution_output, feed_dict = feed_dict)
    print(run_my_convolution_output)

    print('-' * 50)
    print('\nInput = 序列长度 21')
    print('ReLU 激活函数输出（序列长度=21）:')
    run_my_activation_output = sess.run(my_activation_output, feed_dict = feed_dict)
    print(run_my_activation_output)

    print('-' * 50)
    print('\nInput = 序列长度 21')
    print('卷积输出（序列长度=17，滤波器长度=5，步长=1）：')
    run_my_max_pool_output = sess.run(my_max_pool_output, feed_dict = feed_dict)
    print(run_my_max_pool_output)

    print('-' * 50)
    print('\nInput = 序列长度 17')
    print('全连接层输出（序列长度=5）：')
    run_my_full_output = sess.run(my_full_output, feed_dict = feed_dict)
    print(run_my_full_output)

    print('-' * 50)
    print('\n模型输出序列长度=5')
    pass


# ---------------------------------------------------|
# -------------------2D-data-------------------------|
# ---------------------------------------------------|
def convolution_2d_data():
    show_title("2-D 图片数据")

    # Reset Graph
    ops.reset_default_graph()
    sess = tf.Session()

    # Generate 2D data
    data_size = [10, 10]
    data_2d = np.random.normal(size = data_size)

    # --------Placeholder--------
    x_input_2d = tf.placeholder(dtype = tf.float32, shape = data_size)

    # Convolution
    def conv_layer_2d(input_2d, my_filter):
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)
        convolution_output = tf.nn.conv2d(input_4d, filter = my_filter, strides = [1, 2, 2, 1], padding = 'VALID')
        conv_output_2d = tf.squeeze(convolution_output)
        return conv_output_2d

    # Create Convolutional Filter
    my_filter = tf.Variable(tf.random_normal(shape = [2, 2, 1, 1, ]))
    # Create Convolutional Layer
    my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

    # --------Activation--------
    def activation(input_2d):
        return tf.nn.relu(input_2d)

    # Create Activation Layer
    my_activation_output = activation(my_convolution_output)

    # --------Max Pool--------
    def max_pool(input_2d, width, height):
        input_3d = tf.expand_dims(input_2d, 0)
        input_4d = tf.expand_dims(input_3d, 3)
        pool_output = tf.nn.max_pool(
                input_4d, ksize = [1, height, width, 1], strides = [1, 1, 1, 1], padding = 'VALID')
        pool_output_2d = tf.squeeze(pool_output)  # 去除增加的维度
        return pool_output_2d

    # Create Max-Pool Layer
    my_max_pool_output = max_pool(my_activation_output, width = 2, height = 2)

    # --------Fully Connected--------
    def fully_connected(input_layer, num_outputs):
        flat_input = tf.reshape(input_layer, [-1])
        # We then find out how long it is, and create an array for the shape of
        # the multiplication weight = (WxH) by (num_outputs)
        weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
        weight = tf.random_normal(weight_shape, stddev = 0.1)
        bias = tf.random_normal(shape = [num_outputs])
        input_2d = tf.expand_dims(flat_input, 0)
        full_output = input_2d @ weight + bias
        full_output_2d = full_output
        return full_output_2d

    # Create Fully Connected Layer
    my_full_output = fully_connected(my_max_pool_output, 5)

    # Run graph
    # Initialize Variables
    init = tf.global_variables_initializer()
    sess.run(init)

    feed_dict = {x_input_2d: data_2d}

    print('-' * 50)
    print('Input = 输入 [10 X 10] array')
    print('2x2 卷积层, stride size = [2x2], 输出 [5x5] array:')
    print(sess.run(my_convolution_output, feed_dict = feed_dict))

    # Activation Output
    print('\nInput = my_convolution_output [5x5] array')
    print('ReLU 输出 [5x5] array:')
    print(sess.run(my_activation_output, feed_dict = feed_dict))

    # Max Pool Output
    print('\nInput = my_activation_output [5x5] array')
    print('MaxPool, stride size = [1x1], 输出 [4x4] array:')
    print(sess.run(my_max_pool_output, feed_dict = feed_dict))

    # Fully Connected Output
    print('\nInput = my_max_pool_output [4x4] array')
    print('全连接层将 [4x4] 展开成 1D 长度=16，输出长度=5:')
    print(sess.run(my_full_output, feed_dict = feed_dict))
    pass


# -----------------------------------------------------------------
if __name__ == "__main__":
    # convolution_1d_data()
    convolution_2d_data()

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
