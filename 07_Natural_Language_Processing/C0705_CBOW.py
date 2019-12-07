# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0705_CBOW.py
@Version    :   v0.1
@Time       :   2019-12-06 11:29
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0705，P162
@Desc       :   自然语言处理，使用 TensorFlow 实现 CBOW 词嵌入模型
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

from text_tools import build_dictionary, generate_batch_data, load_movie_data, normalize_text, text_to_numbers

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
batch_size = 500
embedding_size = 200
vocabulary_size = 7500
generations = 50000
model_learning_rate = 0.25

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 3  # How many words to consider left and right.

# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

# Declare stop words
stops = stopwords.words('english')

# We pick some test words. We are expecting synonyms to appear
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# Later we will have to transform these into indices

# Load the movie review data
print('Loading Data')
texts, target = load_movie_data()

# Normalize text
print('Normalizing Text Data')
texts = normalize_text(texts, stops)

# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# Build our data set and dictionaries
print('Creating Dictionary')
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]

print('Creating Model')
# Create data/target placeholders
# ToDo:为什么要固定输入数据的大小？
x_inputs = tf.placeholder(tf.int32, shape = [batch_size, 2 * window_size], name = 'x_inputs')
y_target = tf.placeholder(tf.int32, shape = [batch_size, 1], name = 'y_target')
valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# 7. 处理单词嵌套。CBOW 模型将上下文窗口内的单词嵌套叠加在一起
embed = tf.zeros([batch_size, embedding_size])
for element in range(2 * window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, embedding_size], stddev = 1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = y_target,
                                     inputs = embed,
                                     num_sampled = num_sampled,
                                     num_classes = vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = model_learning_rate).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims = True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

# Create model saving operation
saver = tf.train.Saver({"embeddings": embeddings})

# Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)

# Filter out sentences that aren't long enough:
text_data = [x for x in text_data if len(x) >= (2 * window_size + 1)]

# Run the CBOW model.
print('Starting Training')
data_folder_name = 'temp'
loss_vec, loss_x_vec = [], []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size, method = 'cbow')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict = feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict = feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))
        pass

    # Validation: Print some random words and top 5 related words
    if (i + 1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict = feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5  # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
                pass
            print(log_str)

    # Save dictionary + embeddings
    if (i + 1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)
            pass
        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
        pass

    pass

# ----------------------------------------------------------------------
# 运行结束的提醒
winsound.Beep(600, 500)
if len(plt.get_fignums()) != 0:
    plt.show()
pass
