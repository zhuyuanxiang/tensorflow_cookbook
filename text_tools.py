# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   text_tools.py
@Version    :   v0.1
@Time       :   2019-12-06 11:34
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0704，P155
@Desc       :   自然语言处理，使用 TensorFlow 实现 Skip-Gram、CBOW、Doc2Vec的工具函数
@理解：
"""
import os
import string
import sys

import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
from nltk import collections
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

# Load the movie review data
def load_movie_data():
    data_file_path = "../Data"
    pos_file = os.path.join(data_file_path, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(data_file_path, 'rt-polaritydata', 'rt-polarity.neg')

    pos_data = []
    with open(pos_file, 'r', encoding = 'latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors = 'ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding = 'latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors = 'ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data)

    return texts, target


# Normalize text
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in stops]) for x in texts]

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    return texts


# 使用原始文本生成单词表（表中存放的是所有原始文本中出现的单词）
def build_dictionary(sentences, vocabulary_size):
    # 把句子转换成单词列表
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    # count里面存放的内容 [word, word_count]，存放的第一个[单词，单词数]对是[未知单词，-1]
    count = [('RARE', -1)]

    # 使用nltk.collections提供的工具统计单词出现的频率
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # 将单词与索引对应
    # ToKnown：原作者使用单词序列长度累加的方法获得索引，速度慢而且不容易理解，我修改为直接枚举
    word_dict = {x: i for i, (x, _) in enumerate(count)}
    # word_dict = {}
    # for word, word_count in count:
    #     word_dict[word] = len(word_dict)

    return word_dict


# 将句子中的单词转换为字典中的索引
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method = 'skip_gram'):
    # 填充批量数据
    batch_data, label_data = [], []
    while len(batch_data) < batch_size:
        # 随机选择句子
        rand_sentence_ix = int(np.random.choice(len(sentences), size = 1))
        rand_sentence = sentences[rand_sentence_ix]
        # 从句子中根据窗口大小取出一个单词序列
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)]
                            for ix, x in enumerate(rand_sentence)]
        # 根据窗口中序列的定义确定相应的标签索引（即窗口中标签的位置，即中心词的位置）
        label_indices = [ix if ix < window_size else window_size
                         for ix, x in enumerate(window_sequences)]

        # 取出每个窗口的中心词，为每个窗口创建元组
        batch, labels = [], []
        if method == 'skip_gram':
            # x[:y]中心词左边的词；x[(y+1)]中心词右边的词
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]  # （中心词，周围词）
            if len(tuple_data) > 0:
                batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method == 'cbow':
            # x[:y]中心词左边的词；x[(y+1)]中心词右边的词
            batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            # tuple_data = [(x_, y) for x, y in batch_and_labels for x_ in x]
            batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
            # Only keep windows with consistent 2*window_size
            if len(batch_and_labels) > 0:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method == 'doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i + window_size], rand_sentence[i + window_size]) for i in
                                range(0, len(rand_sentence) - window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data
