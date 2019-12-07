# -*- encoding: utf-8 -*-
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0703_implementing_tf_idf.py
@Version    :   v0.1
@Time       :   2019-12-05 16:05
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0703，P149
@Desc       :   自然语言处理，使用 TensorFlow 实现 TF-IDF 算法
@理解：
"""
# common imports
import os
import string
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from sklearn.feature_extraction.text import TfidfVectorizer
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

batch_size = 200
max_features = 8000
# ----------------------------------------------------------------------
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

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# 将标签整数化， 'spam' 表示垃圾短信，设置为1, 'ham' 表示正常短信，设置为0
target = [1 if x == 'spam' else 0 for x in target]

# 文本标准化
texts = [x.lower() for x in texts]  # 文本字母小写
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]  # 移除标点符号
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]  # 移除数字
texts = [' '.join(x.split()) for x in texts]  # 移除多余的空格


# 定义分词器：定义函数的原因是可以加入更多自定义分词内容
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


tfidf = TfidfVectorizer(tokenizer = tokenizer, stop_words = 'english', max_features = max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

sparse_tfidf_text_length = sparse_tfidf_texts.shape[0]
train_indices = np.random.choice(sparse_tfidf_text_length, round(sparse_tfidf_text_length * 0.8), replace = False)
test_indices = np.array(list(set(range(sparse_tfidf_text_length)) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape = [max_features, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Initialize placeholders
x_data = tf.placeholder(shape = [None, max_features], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Declare logistic model (sigmoid in loss function)
model_output = x_data @ A + b

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y_target))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
train_loss, test_loss, train_acc, test_acc = [], [], [], []
i_data = []
for i in range(10001):
    rand_index = np.random.choice(texts_train.shape[0], size = batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    train_feed_dict = {x_data: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict = train_feed_dict)

    # Only record loss and accuracy every 100 generations
    if i % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict = train_feed_dict)
        train_loss.append(train_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict = train_feed_dict)
        train_acc.append(train_acc_temp)

        test_feed_dict = {x_data: texts_test.todense(), y_target: np.transpose([target_test])}
        test_loss_temp = sess.run(loss, feed_dict = test_feed_dict)
        test_loss.append(test_loss_temp)

        test_acc_temp = sess.run(accuracy, feed_dict = test_feed_dict)
        test_acc.append(test_acc_temp)

    if i % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('迭代次数 # {}. 训练集损失（测试集损失）: {:.2f} ({:.2f}). 训练集精度（测试集精度）: {:.2f} ({:.2f})'
              .format(*acc_and_loss))

# Plot loss over time
plt.figure()
plt.plot(i_data, train_loss, 'k-', label = '训练集损失')
plt.plot(i_data, test_loss, 'r--', label = '测试集损失', linewidth = 4)
plt.title('每次迭代的交叉熵损失')
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.legend(loc = 'upper right')

# Plot train and test accuracy
plt.figure()
plt.plot(i_data, train_acc, 'k-', label = '训练集精度')
plt.plot(i_data, test_acc, 'r--', label = '测试集精度', linewidth = 4)
plt.title('训练集精度 和 测试集精度')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.legend(loc = 'lower right')
# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
