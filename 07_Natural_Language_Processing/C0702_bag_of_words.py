# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0702_bag_of_words.py
@Version    :   v0.1
@Time       :   2019-11-07 17:11
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0702，P144
@Desc       :   自然语言处理，使用 TensorFlow 实现“词袋”
@理解：
1. 这个模型是个错误的模型，因为数据集本身就是87%的正常短信，那么只要判断为正常短信就有87%的准确率。
而模型的准确率还不到87%，说明正确理解数据集是非常重要的。
2. 跟踪sess.run(x_col_sums,feed_dict = {x_data: t})，也会发现训练的嵌入矩阵的结果就是UNKNOWN单词和'to'单词过多的短信就是垃圾短信，
这个也是因为数据集中数据偏离造成的，根本原因还是模型与数据不匹配。
"""
# common imports
import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
import winsound
from tensorflow.contrib import learn
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

# 统计文本中不同长度的单词的数目，最大单词长度不超过50个字母
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins = 25)
plt.title("图7-1：文本数据中的单词长度的直方图")

sentence_size = 25  # 每个句子的单词个数最多不超过25个，不足25个用0填充，超过25个的从后往前截断
min_word_freq = 3  # 单词出现的频率不低于3次，如果某个单词只在某几条短信中出现，那么就不选入字典

# TensorFlow 自带的分词器 VocabularyProcessor()
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency = min_word_freq)
# Have to fit transform to get length of unique words.
vocab_processor.fit_transform(texts)  # 使用文本数据进行训练并且变换为字典
embedding_size = len(vocab_processor.vocabulary_)  # 取字典大小为嵌入层的大小

# 将文本数据切分为训练数据集（80%）和测试数据集（20%）
train_indices = np.random.choice(len(texts), int(round(len(texts) * 0.8)), replace = False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

# 设置单位矩阵用于 One-Hot 编码
identity_mat = tf.diag(tf.ones(shape = [embedding_size]))

# 为 logistic regression 创建变量
A = tf.Variable(tf.random_normal(shape = [embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# 初始化占位符
x_data = tf.placeholder(shape = [sentence_size], dtype = tf.int32)
y_target = tf.placeholder(shape = [1, 1], dtype = tf.float32)

# 搜索 Text-Vocab Embedding 权重，单位矩阵用于映射句子中的单词的 One-Hot 向量
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)  # ToDo:为什么要按列求和?

# 模型的输出
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = x_col_sums_2D @ A + b

# 交叉熵损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target, logits = model_output))

# Prediction operation
prediction = tf.sigmoid(model_output)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
print('基于训练集中{}个句子开始训练。。。'.format(len(texts_train)))
loss_vec, train_acc_all, train_acc_avg = [], [], []
for ix, t in enumerate(vocab_processor.transform(texts_train)):  # 只转换不训练，不应该再次训练
    y_data = [[target_train[ix]]]

    sess.run(train_step, feed_dict = {x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict = {x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if ix % 100 == 0:
        print('训练集迭代次数： #' + str(ix + 1) + ': Loss = ' + str(temp_loss))
        pass

    [[temp_pred]] = sess.run(prediction, feed_dict = {x_data: t, y_target: y_data})
    # 获得预测结果
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        # 跟踪最后50个训练精度的平均值
        train_acc_avg.append(np.mean(train_acc_all[-50:]))
        pass
    pass

# 获取测试集的评估精度
print('基于测试集中{}个句子开始评估。。。'.format(len(texts_test)))
test_acc_all = []
for ix, t in enumerate(vocab_processor.transform(texts_test)):
    y_data = [[target_test[ix]]]
    if ix % 50 == 0:
        print("测试集迭代次数 #", ix + 1)
        pass
    [[temp_pred]] = sess.run(prediction, feed_dict = {x_data: t, y_target: y_data})
    test_acc_temp = target_test[ix] == np.round(temp_pred)
    test_acc_all.append(test_acc_temp)
    pass
print("\n测试集精度: {}".format(np.mean(test_acc_all)))

# Plot training accuracy over time
plt.figure()
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'b-', label = "训练集精度")
plt.title("统计最后50个训练集数据的平均训练集精度")
plt.xlabel('迭代次数')
plt.ylabel("训练集精度")

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
