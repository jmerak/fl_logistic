from tensorflow.examples.tutorials.mnist import input_data

'''
MNIST数据集：此处采用tensorflow sample所带的mnist数据集的预处理脚本，input_data.py
实现了数据的读取，向量化。
number of trian data is 55000
number of test data is 10000
每个图片28*28=784维
10分类
'''
# 1.数据读取
'''
one_hot:一种映射编码方式
特征并不总是连续值，而有可能是分类值。比如星期类型，有星期一、星期二、……、星期日
若用[1,7]进行编码，求距离的时候周一和周日距离很远（7），这不合适。
故周一用[1 0 0 0 0 0 0],周日用[0 0 0 0 0 0 1],这就是one-hot编码
对于离散型特征，基于树的方法是不需要使用one-hot编码的，例如随机森林等。
基于距离的模型，都是要使用one-hot编码，例如神经网络等。
'''


def get_dataset(dir):
    minist = input_data.read_data_sets(dir, one_hot=True)
    train_x = minist.train.images
    train_y = minist.train.labels
    test_x = minist.test.images
    test_y = minist.test.labels
    print("----------MNIST loaded----------------")
    print("train shape:", train_x.shape, train_y.shape)
    print("test  shape:", test_x.shape, test_y.shape)

    return train_x, train_y, test_x, test_y
