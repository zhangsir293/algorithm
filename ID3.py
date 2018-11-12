# encoding=utf-8

import cv2
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    """
    cv2.threshold（）
    函数：
    第一个参数:src指原图像，原图像应该是灰度图。
    第二个参数:x指用来对像素值进行分类的阈值。
    第三个参数:y指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    第四个参数:Methods指，不同的不同的阈值方法，这些方法包括：
    •cv2.THRESH_BINARY:大于阈值的像素点的灰度值设定为最大值(如8位灰度值最大为255)，灰度值小于阈值的像素点的灰度值设定为0
    •cv2.THRESH_BINARY_INV:大于阈值的像素点的灰度值设定为0，而小于该阈值的设定为255
    •cv2.THRESH_TRUNC:像素点的灰度值小于阈值不改变，大于阈值的灰度值的像素点就设定为该阈值
    •cv2.THRESH_TOZERO:像素点的灰度值小于该阈值的不进行任何改变，而大于该阈值的部分，其灰度值全部变为0
    •cv2.THRESH_TOZERO_INV像素点的灰度值大于该阈值的不进行任何改变，像素点的灰度值小于该阈值的，其灰度值全部变为0
    """
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


def binaryzation_features(trainset):
    features = []

    for img in trainset:
        img = np.reshape(img, (28, 28))
        """
        ndarray数组内容类型转换
        数据类型	描述
        bool_	以字节存储的布尔值（True 或 False）
        int_	默认的整数类型（和 C 的 long 一样，是 int64 或者 int32）
        intc	和 C 的 int 相同（一般为 int64 或 int32）
        intp	用于下标的整数（和 C 的 ssize_t 相同，一般为int64 或者 int32）
        int8	字节（-128 到 127）
        int16	整数（-32768 到 32767）
        int32	整数（-2147483648 到 2147483647）
        int64	整数（-9223372036854775808 到 9223372036854775807）
        uint8	无符号整数（0 到 255）
        uint16	无符号整数（0 到 65535）
        uint32	无符号整数（0 到 4294967295）
        uint64	无符号整数（0 到 18446744073709551615）
        float_	float64 的简写
        float16	半精度浮点：1位符号，5位指数，10位尾数
        float32	单精度浮点：1位符号，8位指数，23位尾数
        float64	双精度浮点：1位符号，11位指数，52位尾数
        complex_	complex128 的简写
        complex64	由两个32位浮点（实部和虚部）组成的复数
        complex128	由两个64位浮点（实部和虚部）组成的复数
        """
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(img_b)

    features = np.array(features)
    """
    np.reshape(features, (-1, feature_len)),行坐标设置为-1，就是不设置行坐标，行坐标自适应纵坐标数值的变化
    """
    features = np.reshape(features, (-1, feature_len))

    return features


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
        self.node_type = node_type  # 节点类型（internal或leaf）
        self.dict = {}  # dict的键表示特征Ag的可能值ai，值表示根据ai得到的子树
        self.Class = Class  # 叶节点表示的类，若是内部节点则为none
        self.feature = feature  # 表示当前的树即将由第feature个特征划分（即第feature特征是使得当前树中信息增益最大的特征）

    def add_tree(self, key, tree):
        """
        :param key: 划分特征量（0/1）
        :param tree: 字数
        :return: 添加了字数的新树
        """
        self.dict[key] = tree

    def predict(self, features):
        """
        :param features: 要预测数据的特征数组
        :return: 预测得到的分类
        """
        if self.node_type == 'leaf' or (features[self.feature] not in self.dict):
            return self.Class
        #迭代查找所有子树，直到找到叶节点或者发现该特征值不在决策树划分条件中
        tree = self.dict.get(features[self.feature])
        return tree.predict(features)


# 计算数据集x的经验熵H(x)
def calc_ent(x):
    """
    经验熵就是把所有的类别的ent -= p * logp加起来，p是该类别在在样本中出现的概率，即该类别样本数/总样本数
    :param x: 类别（数据标签）
    :return: 经验熵
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


# 计算条件熵H(y/x)
def calc_condition_ent(x, y):
    """
    条件熵H(y/x)=连加含有特征x的类的样本个数float(sub_y.shape[0])/总样本个数 y.shape[0]*含有特征x的类的样本个数的经验熵calc_ent(sub_y)
    :param x: 第x个特征
    :param y: 类别（数据标签）
    :return: 条件熵
    """
    #遍历包含特征x样本
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        #含有特征x子类
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


# 计算信息增益
def calc_ent_grap(x, y):
    """
    :param x: 第x个特征
    :param y: 数据标签（类别）
    :return: 信息增益
    """
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


# ID3算法
def recurse_train(train_set, train_label, features):
    """
    :param train_set: 
    :param train_label: 
    :param features: 
    :return: 
    训练集train_set中的所有实例都属于同一类Ck
    如果特征集features为空
    计算信息增益小于阈值返回生成的决策树，否则继续递归生成决策树
    """
    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果训练集train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果特征集features为空
    class_len = [(i, len(list(filter(lambda x: x == i, train_label)))) for i in range(class_num)]  # 计算每一个类出现的个数
    (max_class, max_len) = max(class_len, key=lambda x: x[1])

    if len(features) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益,并选择信息增益最大的特征
    max_feature = 0
    max_gda = 0
    D = train_label
    for feature in features:
        # print(type(train_set))
        #遍历所有特征，选择当前最佳分类特征
        A = np.array(train_set[:, feature].flat)  # 选择训练集中的第feature列（即第feature个特征)
        gda = calc_ent_grap(A, D)
        if gda > max_gda:
            max_gda, max_feature = gda, feature

    # 步骤4——信息增益小于阈值
    if max_gda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 步骤5——构建非空子集
    sub_features = list(filter(lambda x: x != max_feature, features))
    #Tree第二个属性是节点类型，中间节点记录当前max_feature，叶节点
    tree = Tree(INTERNAL, feature=max_feature)

    max_feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set(
        [max_feature_col[i] for i in range(max_feature_col.shape[0])])  # 保存信息增益最大的特征可能的取值 (shape[0]表示计算行数)
    for feature_value in feature_value_list:

        index = []
        for i in range(len(train_label)):
            #遍历样本，满足特征max_feature,记录索引
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        #递归划分子树
        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features)
        tree.add_tree(feature_value, sub_tree)

    return tree


def train(train_set, train_label, features):
    return recurse_train(train_set, train_label, features)


def predict(test_set, tree):
    """
    :param test_set: 测试数据
    :param tree: 决策树
    :return: 对测试数据的分类结果
    """
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)


class_num = 10  # MINST数据集有10种labels，分别是“0,1,2,3,4,5,6,7,8,9”
feature_len = 784  # MINST数据集每个image有28*28=784个特征（pixels）
epsilon = 0.001  # 设定阈值

if __name__ == '__main__':

    print("Start read data...")

    time_1 = time.time()

    raw_data = pd.read_csv('train.csv', header=0)  # 读取csv数据
    data = raw_data.values

    imgs = data[::, 1::]
    features = binaryzation_features(imgs)  # 图片二值化(很重要，不然预测准确率很低)
    labels = data[::, 0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=0)
    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    # 通过ID3算法生成决策树
    print('Start training...')
    tree = train(train_features, train_labels, list(range(feature_len)))
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = predict(test_features, tree)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    # print("预测的结果为：")
    # print(test_predict)
    for i in range(len(test_predict)):
        if test_predict[i] == None:
            test_predict[i] = epsilon
    print("result:")
    print(test_predict)
    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)
