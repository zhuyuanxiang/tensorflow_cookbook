# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://github.com/zhuyuanxiang/tensorflow_cookbook
---------------------------
@Software   :   PyCharm
@Project    :   TensorFlow_Machine_Learning_Cookbook
@File       :   C0108_data_gathering.py
@Version    :   v0.1
@Time       :   2019-10-29 10:44
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0108，P14
@Desc       :   TensorFlow 基础，载入需要的数据集
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np  # pip install numpy<1.17，小于1.17就不会报错
import sklearn
import tensorflow as tf
from tensorflow.python.framework import ops

# 设置数据显示的精确度为小数点后3位

np.set_printoptions(precision = 8, suppress = True, threshold = np.inf, linewidth = 200)

# 利用随机种子，保证随机数据的稳定性，使得每次随机测试的结果一样
np.random.seed(42)

# 初始化默认的计算图
ops.reset_default_graph()
# Python ≥3.5 is required
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"
# 屏蔽警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Open graph session
sess = tf.Session()


# Data gathering
# -----------------------------------------------------------------
# This function gives us the ways to access
# the various data sets we will need
# 数据已经存放在 Data 中，就不用考验那个痛苦的网络了

# -----------------------------------------------------------------
# Iris Data
def load_iris_data():
    from sklearn import datasets
    # 分类数据
    # 鸢尾花（3种），特征4种（花萼长度、花萼宽度、花瓣长度、花瓣宽度），150条数据
    iris = datasets.load_iris()
    print("iris.data.shape = ", iris.data.shape)
    print(len(iris.data))
    print(len(iris.target))
    print(iris.data[0])
    print(set(iris.target))


# -----------------------------------------------------------------
# Low Birthrate Data
def load_birthate_data():
    import requests

    birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')[5:]
    birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
    birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    print(len(birth_data))
    print(len(birth_data[0]))


# -----------------------------------------------------------------
# Housing Price Data
def load_housing_price_data():
    from sklearn import datasets
    # 回归数据
    # 波士顿房价数据，特征14种，560条数据
    boston_house = datasets.load_boston()
    print("boston_house.data.shape = ", boston_house.data.shape)

    import requests

    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                      'MEDV']
    housing_file = requests.get(housing_url)
    housing_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in housing_file.text.split('\n') if
                    len(y) >= 1]
    print(len(housing_data))
    print(len(housing_data[0]))


# -----------------------------------------------------------------
# MNIST Handwriting Data
def load_mnist_data():
    from sklearn import datasets
    # MINIST 手写体字库
    mnist = datasets.load_digits()
    print(mnist.data.shape)

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("../Data/MNIST_data/", one_hot = True)
    print(len(mnist.train.images))
    print(len(mnist.test.images))
    print(len(mnist.validation.images))
    print(mnist.train.labels[1, :])


# -----------------------------------------------------------------
# Ham/Spam Text Data
def load_spam_ham_text_data():
    # spam：垃圾邮件；ham：非垃圾邮件

    import requests
    import io
    from zipfile import ZipFile

    # Get/read zip file
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors = 'ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]
    [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
    print(len(text_data_train))
    print(set(text_data_target))
    print(text_data_train[1])


# -----------------------------------------------------------------
# Movie Review Data
def load_movie_review_data():
    # 影评（好评、差评）
    # 注：也可以从NLTK中获得
    import requests
    import io
    import tarfile

    movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    r = requests.get(movie_data_url)
    # Stream data into temp object
    stream_data = io.BytesIO(r.content)
    tmp = io.BytesIO()
    while True:
        s = stream_data.read(16384)
        if not s:
            break
        tmp.write(s)
    stream_data.close()
    tmp.seek(0)
    # Extract tar file
    tar_file = tarfile.open(fileobj = tmp, mode = "r:gz")
    pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
    neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
    # Save pos/neg reviews
    pos_data = []
    for line in pos:
        pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors = 'ignore').decode())
    neg_data = []
    for line in neg:
        neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors = 'ignore').decode())
    tar_file.close()

    print(len(pos_data))
    print(len(neg_data))
    print(neg_data[0])


# -----------------------------------------------------------------
# The Works of Shakespeare Data
def load_shakespeare_data():
    # Gutenberg中的Shakespeare的书
    # 注：也可以从NLTK中获得
    import requests

    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    # Get Shakespeare text
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    shakespeare_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    shakespeare_text = shakespeare_text[7675:]
    print(len(shakespeare_text))


# -----------------------------------------------------------------
# English-German Sentence Translation Data
def load_eng_ger_sent_translate_data():
    # 英语——德语句子翻译样本集
    import requests
    import io
    from zipfile import ZipFile

    sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
    r = requests.get(sentence_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('deu.txt')
    # Format Data
    eng_ger_data = file.decode()
    eng_ger_data = eng_ger_data.encode('ascii', errors = 'ignore')
    eng_ger_data = eng_ger_data.decode().split('\n')
    eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
    [english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
    print(len(english_sentence))
    print(len(german_sentence))
    print(eng_ger_data[10])


# -----------------------------------------------------------------
if __name__ == "__main__":
    load_iris_data()

    load_birthate_data()

    load_housing_price_data()

    load_mnist_data()

    load_spam_ham_text_data()

    load_movie_review_data()

    load_shakespeare_data()

    load_eng_ger_sent_translate_data()
    if len(plt.get_fignums()) != 0:
        import winsound

        # 运行结束的提醒
        winsound.Beep(600, 500)
        plt.show()
    pass
