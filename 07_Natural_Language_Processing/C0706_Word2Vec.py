# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0706_Word2Vec.py
@Version    :   v0.1
@Time       :   2019-12-06 16:34
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0706，P167
@Desc       :   自然语言处理，使用 TensorFlow 实现 基于 Word2Vec 的情感分析
@理解：
"""
# common imports
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from nltk.corpus import stopwords
from tensorflow.python.framework import ops

from text_tools import load_movie_data, normalize_text, text_to_numbers

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
# Declare model parameters
batch_size = 100
embedding_size = 200
vocabulary_size = 7500
max_words = 50

# Declare stop words
stops = stopwords.words('english')

# Load the movie review data
print('Loading Data')
texts, target = load_movie_data()

# Normalize text
print('Normalizing Text Data')
texts = normalize_text(texts, stops)

# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# Split up data set into train/test
train_indices = np.random.choice(len(target), round(0.8 * len(target)), replace = False)
test_indices = np.array(list(set(range(len(target))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Load dictionary and embedding matrix
print('加载 CBOW 保存的字典')
dict_file = os.path.join('temp', 'movie_vocab.pkl')
word_dictionary = pickle.load(open(dict_file, 'rb'))

# Convert texts to lists of indices
text_data_train = np.array(text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to specific length
text_data_train = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_test]])

print('Creating Model')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Define model:
# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape = [embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Initialize placeholders
x_data = tf.placeholder(shape = [None, max_words], dtype = tf.int32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Lookup embeddings vectors
embed = tf.nn.embedding_lookup(embeddings, x_data)
# Take average of all word embeddings in documents
embed_avg = tf.reduce_mean(embed, 1)

# Declare logistic model (sigmoid in loss function)
model_output = embed_avg @ A + b

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y_target))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
my_opt = tf.train.AdagradOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Load model embeddings
model_checkpoint_path = os.path.join('temp', 'cbow_movie_embeddings.ckpt')
saver = tf.train.Saver({"embeddings": embeddings})
saver.restore(sess, model_checkpoint_path)

# Start Logistic Regression
print('Starting Model Training')
train_loss, test_loss = [], []
train_acc, test_acc = [], []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size = batch_size)
    rand_x = text_data_train[rand_index]
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict = {x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_loss_temp = sess.run(loss, feed_dict = {x_data: text_data_test, y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        test_acc_temp = sess.run(accuracy, feed_dict = {x_data: text_data_test, y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
        if (i + 1) % 500 == 0:
            acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
            acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
            print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'
                  .format(*acc_and_loss))

# Plot loss over time
plt.figure()
plt.plot(i_data, train_loss, 'k-', label = '训练集')
plt.plot(i_data, test_loss, 'r--', label = '测试集', linewidth = 4)
plt.title('每次迭代的交叉熵损失')
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.legend(loc = 'upper right')

# Plot train and test accuracy
plt.figure()
plt.plot(i_data, train_acc, 'k-', label = '训练集')
plt.plot(i_data, test_acc, 'r--', label = '测试集', linewidth = 4)
plt.title('训练集和测试集的精度')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.legend(loc = 'lower right')
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
