# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0704_skip_gram.py
@Version    :   v0.1
@Time       :   2019-12-05 17:12
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0704，P155
@Desc       :   自然语言处理，使用 TensorFlow 实现 Skip-Gram 模型
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
from nltk.corpus import stopwords
from tensorflow.python.framework import ops

# 设置数据显示的精确度为小数点后3位
from text_tools import build_dictionary, generate_batch_data, load_movie_data, normalize_text, text_to_numbers

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
generations = 100000
print_loss_every = 2000

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 2  # How many words to consider left and right.

# Declare stop words
stops = stopwords.words('english')
texts, target = load_movie_data()
texts = normalize_text(texts, stops)

# 每个句子中单词数目必须多于2个。
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# 建立字典与数据集
# word_dictionary：单词字典，（单词：索引）的数据对
# word_dictionary_rev：反转的单词字典，（索引：单词）的数据对，方便根据索引查询单词
# text_data：将文本转换与索引，方便训练
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# 寻找下面5个单词的同义词
print_valid_every = 5000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
valid_examples = [word_dictionary[x] for x in valid_words]  # 将单词转换成字典索引

# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape = [batch_size])
y_target = tf.placeholder(tf.int32, shape = [batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

# 定义嵌入矩阵（10000，200），用于后面训练
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 从嵌入矩阵中寻找到单词所对应的嵌入向量(200，）
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# 定义 NCE 训练需要的权重和偏移
nce_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size], stddev = 1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# 基于NCE损失函数进行训练，输入的是单词对应的嵌入向量，输出的是标签，对应的随机样本是批量数据的一半，类别是字典大小
# 嵌入矩阵搜索本质是个多分类问题，一般使用 softmax 损失函数来解决，但是因为10000个分类导致结果稀疏性太高，而影响模型收敛
# 因此使用噪声对比损失函数（NCE，Noise-Contrastive Error）将问题转换成一个二值预测问题，
# 系统先将 num_samples 个批量数据转换为随机噪声，然后再通过区别真实数据和噪声的方式求解嵌入矩阵。
loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = y_target,
                                     inputs = embed,
                                     num_sampled = num_sampled,
                                     num_classes = vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(loss)

# 寻找相似的单词（使用cosine距离进行度量）
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims = True))
normalized_embeddings = embeddings / norm  # 将训练得到的嵌入向量矩阵归一化，方便进行距离计算
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)  # 搜索关注的单词对应的归一化后的嵌入向量
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)  # 计算关注的单词与字典中所有单词之间的互信息

# Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)

# 运行Skip-Gram模型
loss_vec, loss_x_vec = [], []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict = feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict = feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))

    # Validation: Print some random words and top 5 related words
    if (i + 1) % print_valid_every == 0:
        # 计算关注的单词与字典中的单词的相似度度量矩阵
        sim = sess.run(similarity, feed_dict = feed_dict)
        for j in range(len(valid_words)):
            valid_word = valid_words[j]
            # top_k = number of nearest neighbors
            top_k = 5  # 取出前5个相似度最高的单词的索引
            nearest = (-sim[j, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
                pass
            print(log_str)
            pass
        pass
    pass

# ----------------------------------------------------------------------
# 运行结束的提醒
#
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
