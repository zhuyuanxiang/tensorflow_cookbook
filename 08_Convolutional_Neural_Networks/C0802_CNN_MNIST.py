# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0802_CNN_MNIST.py
@Version    :   v0.1
@Time       :   2019-12-07 10:47
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec08，P182
@Desc       :   卷积神经网络，使用 TensorFlow 实现 简单的 CNN　基于 MNIST 数据集 解决多分类问题
@理解：
"""
# common imports
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)
# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
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
# ----------------------------------------------------------------------
print("加载MNIST数据集，对预测值不使用 One-Hot 编码")
mnist = input_data.read_data_sets("../Data/MNIST_data/", one_hot = False)

# 将数据集中每条数据从形状（1，784）转换成（28，28），方便使用 CNN
train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])

# 标签数据是原始标记（0~9），不是 One-Hot 编码，
# 所以还需要将 softmax 输出转换为标签数字后，才能计算精度
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Set model parameters
batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = np.max(train_labels) + 1
num_channels = 1  # greyscale = 1 channel
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2  # NxN window for 1st max pool layer（窗口大小）
max_pool_size2 = 2  # NxN window for 2nd max pool layer（窗口大小）
fully_connected_size1 = 100

# Declare model placeholders
x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape = x_input_shape)
y_target = tf.placeholder(tf.int32, shape = (batch_size,))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape = eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape = (evaluation_size,))

# Declare model parameters
conv1_weight = tf.Variable(tf.truncated_normal(
        [4, 4, num_channels, conv1_features], stddev = 0.1, dtype = tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype = tf.float32))

conv2_weight = tf.Variable(tf.truncated_normal(
        [4, 4, conv1_features, conv2_features], stddev = 0.1, dtype = tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype = tf.float32))

# fully connected variables
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)

full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight = tf.Variable(tf.truncated_normal(
        [full1_input_size, fully_connected_size1], stddev = 0.1, dtype = tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev = 0.1, dtype = tf.float32))

full2_weight = tf.Variable(tf.truncated_normal(
        [fully_connected_size1, target_size], stddev = 0.1, dtype = tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev = 0.1, dtype = tf.float32))


# Initialize Model Operations
def my_conv_net(conv_input_data):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv2d(conv_input_data, conv1_weight,
                         strides = [1, 1, 1, 1],
                         padding = 'SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1,
                               ksize = [1, max_pool_size1, max_pool_size1, 1],
                               strides = [1, max_pool_size1, max_pool_size1, 1],
                               padding = 'SAME')

    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv2d(max_pool1, conv2_weight,
                         strides = [1, 1, 1, 1],
                         padding = 'SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2,
                               ksize = [1, max_pool_size2, max_pool_size2, 1],
                               strides = [1, max_pool_size2, max_pool_size2, 1],
                               padding = 'SAME')

    # Transform Output into a 1xN layer for next fully connected layer
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])
    # tf.layers.flatten

    # First Fully Connected Layer
    fully_connected1 = tf.nn.relu(flat_output @ full1_weight + full1_bias)

    # Second Fully Connected Layer
    final_model_output = fully_connected1 @ full2_weight + full2_bias

    return final_model_output


# 建立两个神经网络是为了使用不同的值来定义批量大小和评估大小
# 两个神经网络使用的参数是同一个，也就是训练模型网络训练好的参数直接供测试模型网络使用
model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

# Declare Loss Function (softmax cross entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model_output, labels = y_target))

# Create a prediction function
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

# Create an optimizer
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)


# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis = 1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct / batch_predictions.shape[0]


# Start training loop
train_loss = []
train_acc, test_acc = [], []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata), size = batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}

    sess.run(train_step, feed_dict = train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict = train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if (i + 1) % eval_every == 0:
        eval_index = np.random.choice(len(test_xdata), size = evaluation_size)
        eval_x = test_xdata[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict = test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)

        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.figure()
plt.plot(eval_indices, train_loss, 'k-')
plt.title('每次迭代的 Softmax 损失')
plt.xlabel('迭代次数')
plt.ylabel('Softmax 损失')

# Plot train and test accuracy
plt.figure()
plt.plot(eval_indices, test_acc, 'r--', label = '测试集')
plt.title('训练集和测试集的精度')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.legend(loc = 'lower right')

# 绘制最后6张图的预测结果
images = np.squeeze(rand_x[0:6])
actuals = rand_y[0:6]
predictions = np.argmax(temp_train_preds, axis = 1)[0:6]

Nrows = 2
Ncols = 3
for i in range(6):
    plt.subplot(Nrows, Ncols, i + 1)
    plt.imshow(np.reshape(images[i], [28, 28]), cmap = 'Greys_r')
    plt.title('真实值: ' + str(actuals[i]) + ' 预测值: ' + str(predictions[i]), fontsize = 10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
