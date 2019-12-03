代码说明：与作者的原始代码进行合并，这个代码既支持第一版，也支持第二版。

学习建议：书中代码只适合用来了解 TensorFlow ，没有对算法的进行介绍及算法与代码的关联，
因此不建议使用本书来学习机器学习的相关概念。

需要的理论基础：
1. 机器学习算法的基本理解、公式推导（建议先学习周志华《机器学习》和李航《统计学习方法》）
2. Python 的基本语法
3. Scikit-Learn 工具包的基本功能（建议先学习《Python机器学习基础教程》）
4. Numpy 的基本功能

需要的软件开发包：
- numpy 
- tensorflow
- Pillow
- scikit-learn
- matplotlib
    - matplotlib绘图需要提供中文字体支持。
        1. 将 “\Tools” 目录下的 “YaHei.Consolas.1.12.ttf” 文件拷贝到 “\Lib\site-packages\matplotlib\mpl-data\fonts\ttf” 目录下。
        2. 将 “\Tools” 目录下的 “matplotlibrc” 文件拷贝到 “\Lib\site-packages\matplotlib\mpl-data\” 目录下，
        拷贝过程中可以直接覆盖原始文件，也可以将原始文件改名。

[作者的原始代码](https://github.com/nfmcclure/tensorflow_cookbook)

# Ch06 神经网络算法

## 6.1 神经网络算法基础

常用的学习资料：
- Yann Lecun, etc., Efficient BackProp, [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- Stanford CS231, [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu)
- Stanford CS224d, [Deep Learning for Natural Language Processing](http://cs231n.stanford.edu)
- Goodfellow, etc., [Deep Learning](http://www.deeplearningbook.org)
- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
- Andrej Karpathy, [A Hacker's Guid to Neural Networks](http://karpathy.github.io/neuralnets/) 
- Goodfellow, etc., [Deep Learning for Beginners](http://randomekek.github.io/deep/deeplearning.html)

