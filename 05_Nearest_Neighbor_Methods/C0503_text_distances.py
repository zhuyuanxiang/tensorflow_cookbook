# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0503_text_distances.py
@Version    :   v0.1
@Time       :   2019-11-03 12:25
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0503，P95
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 文本距离度量
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

from tools import show_values

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)
# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)

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


# ----------------------------------
# First compute the edit distance between 'bear' and 'beers'
# 字符串间的编辑距离（Levenshtein 距离，最小编辑距离）：由一个字符串转换成另一个字符串所需要的最少编辑操作次数。
# 允许的编辑操作包括：❶插入一个字符，❷删除一个字符，❸替换一个字符
# tf.edit_distance()：计算最小编辑距离
# tf.SparseTensor()：创建稀疏张量，指定字符索引(indices)、张量中的值(values)、矩阵形状(dense_shape)
# dense_shape：好像作用不大，只要有一个是正确的，就可以正确输出
# normal=False：表示计算总的编辑距离
# normal=True：表示计算归一化的编辑距离，通过编辑距离除以第二个单词的长度进行归一化
def compare_one_word():
    truth = list('beers')
    t1 = tf.SparseTensor(indices = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
                         values = truth,
                         dense_shape = [1, 1, 1])  # dense_shape=[多少行数据，每行数据中的单词个数，最长的单词长度]
    show_values(t1, "t1")

    hypothesis = list('bear')
    h1 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
                         hypothesis,
                         [1, 1, 1])
    show_values(tf.edit_distance(h1, t1, normalize = False), "{}与{}的文本距离：".format(hypothesis, truth))

    hypothesis = list('tensor')
    h1 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]],
                         hypothesis,
                         [1, 1, 1])
    show_values(tf.edit_distance(h1, t1, normalize = False), "{}与{}的文本距离：".format(hypothesis, truth))

    hypothesis = list('internet')
    h1 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7]],
                         hypothesis,
                         [1, 1, 1])
    show_values(tf.edit_distance(h1, t1, normalize = False), "{}与{}的文本距离：".format(hypothesis, truth))

    show_values(tf.edit_distance(h1, t1, normalize = False),
                "{}与{}的文本距离：".format(hypothesis, truth))


# ----------------------------------
# Compute the edit distance between ('bear','beer') and 'beers':
# tf.edit_distance(normalize=True)：将数值归一化
def compare_two_words():
    hypothesis2 = list('bearbeer')
    h2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
                          [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]],
                         hypothesis2,
                         [1, 2, 4])

    truth2 = list('beersbeers')
    t2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4],
                          [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4]],
                         truth2,
                         [1, 2, 5])

    show_values(tf.edit_distance(h2, t2, normalize = True), " ('bear','beer')与'beers'的文本距离：")


def compare_two_words_with_space():
    hypothesis2 = list('bear beer')
    truth2 = list('beers beers')
    h2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3],
                          [0, 1, 0],
                          [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3]],
                         hypothesis2,
                         [1, 3, 4])

    t2 = tf.SparseTensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4],
                          [0, 1, 0],
                          [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4]],
                         truth2,
                         [1, 3, 5])

    show_values(tf.edit_distance(h2, t2, normalize = True), " ('bear','beer')与'beers'的文本距离：")
    pass


# ----------------------------------
# Now compute distance between four words and 'beers' more efficiently:
# 书上的结果是错的。(P97)
def compare_word_list():
    hypothesis_words = ['bear', 'bar', 'tensor', 'flow', 'internet']
    truth_word = ['beers']

    num_h_words = len(hypothesis_words)
    h_indices = [[xi, 0, yi] for xi, x in enumerate(hypothesis_words) for yi, y in enumerate(x)]
    h_chars = list(''.join(hypothesis_words))

    h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words, 1, 1])

    truth_word_vec = truth_word * num_h_words
    t_indices = [[xi, 0, yi] for xi, x in enumerate(truth_word_vec) for yi, y in enumerate(x)]
    t_chars = list(''.join(truth_word_vec))

    t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words, 1, 1])

    h3_content = sess.run(h3)
    print("h3_content =", h3_content)
    t3_content = sess.run(t3)
    print("t3_content =", t3_content)

    show_values(tf.edit_distance(h3, t3, normalize = True), "{} 与 {}的文本距离：".format(hypothesis_words, truth_word))
    pass


# ----------------------------------
# 8. 用占位符来计算两个单词列表间的编辑距离
# 因为占位符需要传入值，所以使用 SparseTensorValue() 代替先前的稀疏张量，将数据转换为稀疏张量值用于传入
def compare_two_word_with_placeholders():
    # Create input data
    hypothesis_words = ['bear', 'bar', 'tensor', 'flow']
    truth_word = ['beers']

    def create_sparse_vec(word_list):
        num_words = len(word_list)
        indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
        chars = list(''.join(word_list))
        return tf.SparseTensorValue(indices, chars, [num_words, 1, 1])

    hyp_string_sparse = create_sparse_vec(hypothesis_words)
    truth_string_sparse = create_sparse_vec(truth_word * len(hypothesis_words))

    print("hyp_string_sparse =", hyp_string_sparse)
    print("truth_string_sparse =", truth_string_sparse)

    hyp_input = tf.sparse_placeholder(dtype = tf.string)
    truth_input = tf.sparse_placeholder(dtype = tf.string)

    edit_distances = tf.edit_distance(hyp_input, truth_input, normalize = True)
    feed_dict = {hyp_input: hyp_string_sparse, truth_input: truth_string_sparse}
    show_values(edit_distances, title = "使用 sparse_placeholder 比较单词", feed_dict = feed_dict)
    pass


# 其他距离公式
# 1. 汉明距离（Hamming Distance）：两个等长字符串中对应位置的不同字符的个数
# 2. 余弦距离（Cosine Distance）：不同k-gram的点积除以不同k-gram的L2范数
# 3. 杰卡德距离（Jaccard Distance）：两个字符串中相同字符数除以所有字符数

if __name__ == "__main__":
    # compare_one_word()
    # compare_two_words()
    compare_two_words_with_space()

    # compare_word_list()
    # compare_two_word_with_placeholders()

    # -----------------------------------------------------------------
    # 运行结束的提醒
    winsound.Beep(600, 500)
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass
