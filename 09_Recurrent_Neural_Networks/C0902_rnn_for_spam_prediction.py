# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0902_rnn_for_spam_prediction.py
@Version    :   v0.1
@Time       :   2019-12-09 16:30
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec09，P2
@Desc       :   递归神经网络，使用 TensorFlow 实现 RNN 模型进行垃圾短信预测
@理解：C0703使用TF-IDF+进行垃圾短信预测，本节使用RNN进行垃圾短信预测，可以对比两种效果。
明显比C0703的速度快，但是和C0703一样没有超过基准精度（数据本身是偏的），因此模型都存在问题。
个人感觉作者8章和9章的内容都存在模型问题，对于帮助理解TensorFlow和机器学习帮助不大，也不打算花精力去修改作者的代码。
故打算放弃第9章。
"""
# common imports
import os
import re
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
# Set RNN parameters
epochs = 20
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

# Download or open data
print("载入数据。。。")
# 下载的文件直接读出，没有下载的文件就下载后读出
data_file_name = "../Data/SMS_SPam/SMSSpamCollection"
with open(data_file_name, encoding = 'utf-8') as temp_output_file:
    text_data = temp_output_file.read()
    pass
pass

# Format Data
text_data = text_data.encode('ascii', errors = 'ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sequence_length, min_frequency = min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

# Define the RNN cell
# tensorflow change >= 1.0, rnn is put into tensorflow.contrib directory. Prior version not test.
if tf.__version__[0] >= '1':
    # from tensorflow.contrib.rnn import BasicRNNCell
    # cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
    cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype = tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev = 0.1))
bias = tf.Variable(tf.constant(0.1, shape = [2]))
logits_out = tf.matmul(last, weight) + bias

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_out, labels = y_output)
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss, test_loss = [], []
train_accuracy, test_accuracy = [], []
train_dict = {}
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train) / batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]

        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5}
        sess.run(train_step, feed_dict = train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict = train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict = test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

# Plot loss over time
plt.figure()
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k-', label = '训练集损失')
plt.plot(epoch_seq, test_loss, 'r--', label = '测试集损失', linewidth = 4)
plt.title('训练集 和 测试集 的 Softmax 损失')
plt.xlabel('迭代次数')
plt.ylabel('Softmax 损失')
plt.legend(loc = 'upper left')

# Plot accuracy over time
plt.figure()
plt.plot(epoch_seq, train_accuracy, label = '训练集精度')
plt.plot(epoch_seq, test_accuracy, label = '测试集精度', linewidth = 4)
plt.title('训练集精度 和 测试集精度')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.legend(loc = 'upper left')
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
