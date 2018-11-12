#-*- coding: utf-8 -*-
import pandas as pd
import time

def ToD(d):
    """
    :param 读入的原始数据: 
    :return: 将原始数据转化成DataFrame数据
    最初的原始数据，一行是一个样本，样本里的每一个值是一个特征，
    先将每一个样本转化成一个序列，这个样本含有某项特征，就将该特征的值赋1，之后将所有的样本序列转成DataFrame多维数组，
    每一行是一个样本，每一列是一个特征，有特征，赋1，没有，赋0。
    """
    print('转换原始数据至0-1矩阵')
    start = time.clock()
    """Series赋值，前面列表是内容，后面是索引"""
    ct = lambda x:pd.Series(1,index=x)
    """
    DataFrame.as_matrix()返回一个numpy数组，等价于DataFrame.values，
    将读到的数据做序列化数据的索引，值赋为1。
    """
    b = map(ct,d.as_matrix())
    d = pd.DataFrame(list(b)).fillna(0)
    d = (d==1)
    end  = time.clock()
    #print(d)
    print('转换完毕，用时：%0.2f秒'%(end-start))
    return d

#自定义连接函数，用于实现L_(k-1)到C_k的连接
def conn_string(x,ms):
    """
    :param x:DataFrame索引，即满足支持度要求的项 
    :param ms:连接符：—— 
    :return: 排序后的列表
    """
    x=list(map(lambda i:sorted(i.split(ms)),x))
    #print(x)
    l=len(x[0])
    #print(l)
    r=[]
    for i in range(len(x)):
        for j in range(i,len(x)):
            """
            第i个为连接的元素的除了最后一个特征之外全部相等，对应子集定理,
            这里不存在支持度的计算，只是简单的聚合。以第一个未连接的数据项长度为基准
            """
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
    return r

#寻找关联规则函数
def find_rule(d,support,confidence):
    """
    :param d:DataFrame化后的样本数据 
    :param support: 支持度
    :param confidence: 置信度
    :return: 横轴为关联特征，纵轴为支持度，置信度的二维列表
    """
    start = time.clock()
    result = pd.DataFrame(index=['support','confidence'])

    """
    DataFrame.sum()
    纵向汇总（行求和） aixs=1为横向汇总
    返回值是指为1的特征的统计值构成的series序列
    A1    244
    A2    355
    A3    278
    A4     53
    B1    325
    B2    397
    B3    179
    ······
    """
    support_series = 1.0*d.sum()/len(d)#支持度序列，len(DataFrame),结果是行数，即样本个数
    """
    Series中的值是可以拿来与数值直接比较的，放在索引项，
    可以直接将比较结果为真的项目挑出来生成一个新的序列
    """
    column = list(support_series[support_series > support].index)#初步根据支持度筛选
    k = 0

    while len(column) > 1:
        k = k+1
        print('正在进行第%s次搜索...'%k)
        column = conn_string(column,ms)
        print('候选集数目：%s...'%len(column))
        #print(support_series)
        """
        prod()，axis=0，返回纵向乘积，axis=1返回横向乘积，
        numeric_only=true,值计算数值。
        由于连接函数只是简单的聚合各项，所以连接之后，
        对返回的数据项检查看是否同在一个样本里，有特征为1，没有为0.乘积为1，
        说明样本全部拥有这个聚合里的特征。
        """
        # for each in column:
        #     #print(d[each])
        #     print(d[each].prod(axis=1,numeric_only=True))
        #     print("==========================================")
        sf = lambda i:d[i].prod(axis=1,numeric_only=True)#生成新的候选矩阵

        #创建连接数据，这一步耗时，耗内存最严重，当数据集较大是，可以考虑并行计算优化
        """
        join()函数
        语法：  'sep'.join(seq)
        参数说明
        sep：分隔符。可以为空
        seq：要连接的元素序列、字符串、元组、字典
        上面的语法即：以sep作为分隔符，将seq所有的元素合并成一个新的字符串
        返回值：返回一个以分隔符sep连接各个元素后生成的字符串
        这边所有是特征，所有为了还原成最初计算所用的矩阵，需要转置
        """
        d_2 = pd.DataFrame(list(map(sf,column)),index=[ms.join(i) for i in column]).T

        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        #序列可以像列表一样使用append函数添加元素
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B，原因参考置信度定义公式？
            # print(column)
            # print("====================================")
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])
                # print(column2)
                # print('==============')

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            """
            连接之前的支持度：support_series[ms.join(sorted(i))]
            连接之后的支持度：support_series[ms.join(i[:len(i) - 1])，每次连接多一个元素
            """
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    end = time.clock()
    print('搜索完成，用时：%0.2f秒' % (end - start))
    print('结果为：')
    print(result)

    return result


#
if __name__ == "__main__":
    ms = '--'  # 连接符，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

    # ------------ 官方 ------------
    """如果没有列名，header=0，也可以用一个列表来指定列名，dtype指定每列数据的类型"""
    d = pd.read_csv('apriori.txt', header=None, dtype=object)
    d = ToD(d)
    support = 0.06  # 最小支持度
    confidence = 0.75  # 最小置信度
    output = find_rule(d, support, confidence)
    output.to_excel('rules.xls')

    # ------------自己 ------------
    data = pd.read_csv('CoOccurrence_data_800.csv', header=None)
    support = 0.002  # 最小支持度
    confidence = 0.0  # 最小置信度
    d = ToD(data[[1, 2]])
    output = find_rule(d, support, confidence)


