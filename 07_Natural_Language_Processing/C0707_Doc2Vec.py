# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tensorflow_cookbook
@File       :   C0707_Doc2Vec.py
@Version    :   v0.1
@Time       :   2019-12-06 17:12
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《TensorFlow机器学习实战指南，Nick McClure》, Sec0707，P172
@Desc       :   自然语言处理，使用 TensorFlow 实现基于 Doc2Vec 的情感分析
@理解：关键是文档嵌套与单词嵌套的结合。
结合有两种方式：❶ 文档嵌套和单词嵌套相加；❷ 文档嵌套直接在单词嵌套后面。
这个模型采用的是第2种方式，但是使用的数据集对于理解Doc2Vec方法效果不太好，通过这个例子只能知道如何使用，无法知道这个模型带来的改变是什么。
这个例子还说明，虽然使用神经网络训练不需要考虑太多前期工作，但是前期数据特征化依然是非常重要的，只有对模型的充分理解才能更好的特征化。
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
data_folder_name = 'temp'
batch_size = 500
vocabulary_size = 7500
generations = 100000
model_learning_rate = 0.001

embedding_size = 200  # Word embedding size
doc_embedding_size = 100  # Document embedding size
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 3  # How many words to consider to the left.

# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

# Declare stop words
stops = stopwords.words('english')

# We pick a few test words for validation.
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']
# Later we will have to transform these into indices

# Load the movie review data
print('Loading Data')
texts, target = load_movie_data()

# Normalize text
print('Normalizing Text Data')
texts = normalize_text(texts, stops)

# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
texts = [x for x in texts if len(x.split()) > window_size]
assert (len(target) == len(texts))

# Build our data set and dictionaries
print('Creating Dictionary')
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# 获得检验用的单词的键值
valid_examples = [word_dictionary[x] for x in valid_words]

print('Creating Model')
# 6. 定义单词嵌套，声明对比噪声损失函数（NCE）
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal(
        [vocabulary_size, concatenated_size], stddev = 1.0 / np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape = [None, window_size + 1])  # windows_size是单词嵌套，后面的1是文档嵌套
y_target = tf.placeholder(tf.int32, shape = [None, 1])
valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

# 8. 创建单词嵌套函数和文档嵌套函数，将单词嵌套求和，再与文档嵌套连接在一起
# 创建单词嵌套函数（基于的CBOW方法）
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
# 创建文档嵌套函数（文档索引基于文档导入时顺序的唯一索引值）
doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
# 单词嵌套与文档嵌套的连接
final_embed = tf.concat(axis = 1, values = [embed, tf.squeeze(doc_embed)])

# 9. 声明损失函数和优化器
# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                     biases = nce_biases,
                                     labels = y_target,
                                     inputs = final_embed,
                                     num_sampled = num_sampled,
                                     num_classes = vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = model_learning_rate)
train_step = optimizer.minimize(loss)

# 10. 声明验证单词集的余弦距离
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b = True)

# 11. 创建模型的 Saver 函数，用于保存单词嵌套和文档嵌套
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

# Add variable initializer.
init = tf.global_variables_initializer()
sess.run(init)

# 训练 Doc2Vec 模型
print('Starting Training')
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size, method = 'doc2vec')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(train_step, feed_dict = feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict = feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print('Loss at step {} : {}'.format(i + 1, loss_val))

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
            print(log_str)

    # Save dictionary + embeddings
    if (i + 1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'doc2vec_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))

# Start logistic model-------------------------
# 使用这些嵌套矩阵训练逻辑回归模型
max_words = 20
logistic_batch_size = 500

# Split dataset into train and test sets
# Need to keep the indices sorted to keep track of document index
train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace = False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Convert texts to lists of indices
text_data_train = np.array(text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to specific length
text_data_train = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_test]])

# Define Logistic placeholders
log_x_inputs = tf.placeholder(tf.int32, shape = [None, max_words + 1])  # plus 1 for doc index
log_y_target = tf.placeholder(tf.int32, shape = [None, 1])

# Define logistic embedding lookup (needed if we have two different batch sizes)
# Add together element embeddings in window:
log_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
    log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:, element])

log_doc_indices = tf.slice(log_x_inputs, [0, max_words], [logistic_batch_size, 1])
log_doc_embed = tf.nn.embedding_lookup(doc_embeddings, log_doc_indices)

# concatenate embeddings
log_final_embed = tf.concat(axis = 1, values = [log_embed, tf.squeeze(log_doc_embed)])

# Define model:
# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape = [concatenated_size, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(log_final_embed, A), b)

# Declare loss function (Cross Entropy loss)
logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = model_output, labels = tf.cast(log_y_target, tf.float32)))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(log_y_target, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Declare optimizer
logistic_opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
logistic_train_step = logistic_opt.minimize(logistic_loss, var_list = [A, b])

# Intitialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Start Logistic Regression
print('Starting Logistic Doc2Vec Model Training')
train_loss, test_loss = [], []
train_acc, test_acc = [], []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size = logistic_batch_size)
    rand_x = text_data_train[rand_index]
    # Append review index at the end of text data
    rand_x_doc_indices = train_indices[rand_index]
    # 这里才把输入数据补齐（单词索引+文档索引）
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])

    feed_dict = {log_x_inputs: rand_x, log_y_target: rand_y}
    sess.run(logistic_train_step, feed_dict = feed_dict)

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        rand_index_test = np.random.choice(text_data_test.shape[0], size = logistic_batch_size)
        rand_x_test = text_data_test[rand_index_test]
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
        rand_y_test = np.transpose([target_test[rand_index_test]])

        test_feed_dict = {log_x_inputs: rand_x_test, log_y_target: rand_y_test}

        i_data.append(i + 1)

        train_loss_temp = sess.run(logistic_loss, feed_dict = feed_dict)
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(logistic_loss, feed_dict = test_feed_dict)
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict = feed_dict)
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict = test_feed_dict)
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
