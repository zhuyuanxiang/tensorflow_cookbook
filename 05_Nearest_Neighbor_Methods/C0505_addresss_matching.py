# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0505_addresss_matching.py
@Version    :   v0.1
@Time       :   2019-11-03 17:16
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0505，P101
@Desc       :   基于 TensorFlow 的线性回归，使用 TensorFlow 实现 地址匹配
"""
import os
import random
import string
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

# Address Matching with k-Nearest Neighbors
# ----------------------------------
#
# 对两个数据集进行地址匹配
# 对每个错误的数据返回一个最接近的地址
#
# We will consider two distance functions:
# 1) 对街道地址使用最小编辑距离匹配
# 2) 对邮政编码使用 Euclidian 距离 (L2) 匹配

# First we generate the data sets we will need
# n = Size of created data sets
n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']
rand_zips = [random.randint(65000, 65999) for i in range(5)]


# Function to randomly create one typo in a string w/ a probability
# 3. 随机改变一个字母，用于生成测试集数据的函数
def create_typo(s, prob = 0.75):
    if random.uniform(0, 1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
        pass
    return s


# Generate the reference dataset
# 2. 生成标准数据集（全部是正确数据）
numbers = [random.randint(1, 9999) for i in range(n)]
streets = [random.choice(street_names) for i in range(n)]
street_suffs = [random.choice(street_types) for i in range(n)]
zips = [random.choice(rand_zips) for i in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets, zips)]

# Generate test dataset with some typos
# 3. 生成测试数据集（里面有许多错误数据）
typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

# Now we can perform address matching
# Placeholders
ref_address = tf.sparse_placeholder(dtype = tf.string)
ref_zip = tf.placeholder(shape = [None, n], dtype = tf.float32)
test_address = tf.sparse_placeholder(dtype = tf.string)
test_zip = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Declare Zip code distance for a test zip and reference set
# 5. 声明邮政编码距离 = L2范数
zip_dist = tf.square(ref_zip - test_zip)

# Declare Edit distance for address
# 声明地址距离 = 最小编辑距离
address_distance = tf.edit_distance(test_address, ref_address, normalize = True)

# Create similarity scores
# 6. 把邮政编码距离和地址距离转换成相似度。
# 当两个输入完全一致时相似度为1；完全不一致时相似度为0.
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
zip_sim = (zip_max - zip_dist) / (zip_max - zip_min)
address_sim = 1. - address_distance

# Combine distance functions
# 7. 数据相似度 = 权重*相似度
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.transpose(address_weight * address_sim) + zip_weight * zip_sim

# Predict: Get max similarity entry
# 取出最为相似的数据
top_match_index = tf.argmax(weighted_sim, 1)


# Function to Create a character-sparse tensor from strings
# 为了能够在 TensorFlow 中使用编辑距离，需要把地址字符串转换成稀疏向量
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_vec) for yi, y in enumerate(x)]
    chars = list(''.join(word_vec))
    return tf.SparseTensorValue(indices, chars, [num_words, 1, 1])


# Loop through test indices
# 从参考数据集中提取出地址数据和邮政编码数据
reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# Create sparse address reference set
# 将地址数据转换成稀疏向量
sparse_ref_set = sparse_from_word_vec(reference_addresses)

for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]
    [[test_zip_]] = test_zip_entry

    # Create sparse address vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)

    feed_dict = {
            test_address: sparse_test_set,
            test_zip: test_zip_entry,
            ref_address: sparse_ref_set,
            ref_zip: reference_zips
    }
    best_match = sess.run(top_match_index, feed_dict = feed_dict)
    best_street = reference_addresses[best_match[0]]
    [best_zip] = reference_zips[0][best_match]
    print('-' * 50)
    print("第 {} 个数据：".format(i))
    print("Address: {} , {}".format(test_address_entry, test_zip_))
    print("Match  : {} , {}".format(best_street, best_zip))

# -----------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
