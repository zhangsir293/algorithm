# encoding=utf-8
"""
数据文件train.csv和test.csv包含从0到9的手绘数字的灰度图像。

每个图像的高度为28个像素，宽度为28个像素，总共为784个像素。每个像素具有与其相关联的单个像素值，指示该像素的亮度或暗度，
较高的数字意味着较暗。该像素值是0到255之间的整数，包括0和255。

训练数据集（train.csv）有785列。第一列称为“标签”，是用户绘制的数字。其余列包含关联图像的像素值。

训练集中的每个像素列都具有像pixelx这样的名称，其中x是0到783之间的整数，包括0和783。为了在图像上定位该像素，
假设我们已经将x分解为x = i * 28 + j，其中i和j是0到27之间的整数，包括0和27。然后，pixelx位于28 x 28矩阵的第i行和第j列上（索引为零）。

例如，pixel31表示从左边开始的第四列中的像素，以及从顶部开始的第二行。
"""
"""
sklearn 的决策树实现了cart算法
"""
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    print("读取数据。。。")
    start = time.time()
    """
    pd.read_csv()
    header : int or list of ints, default ‘infer’
    指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，
    否则设置为None。如果明确设定header=0 就会替换掉原来存在列名。
    header参数可以是一个list例如：[0,1,3]，
    这个list表示将文件中的这些行作为列标题（意味着每一列有多个标题），
    介于中间的行将被忽略掉（例如本例中的2；本例中的数据1,2,4行将被作为多级标题出现，
    第3行数据将被丢弃，dataframe的数据从第5行开始。）。
    注意：如果skip_blank_lines=True 那么header参数忽略注释行和空行，所以header=0表示第一行数据而不是文件的第一行。
    """
    raw_data = pd.read_csv('train.csv',header=0)
    data = raw_data.values

    """
    对于data[x,y],x是对数组中的列表，x是第x个列表，y列表中第y个元素
    关于数字索引：
    x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
    r1=np.array([[0,1]])
    r2=np.array([[3,3]])
    rows = np.array([[0,1],[3,3]])
    cols = np.array([[0,2],[0,2]]) 
    #得到的数组坐标[[(0，0),(1,2)\n],[(3,0),(3,2)\n]
    y = x[rows,cols]
    关于数组切片：
    data[x:y:z,a:b:c]
    x:y:z是针对数组中的列表（行）
    a:b:c是针对列表中的元素的（列）
    格式：开始切片位置x：结束切片位置y：步长z
    """
    features = data[::,1::]
    labels = data[::,0]

    #随机选取三分之一的数据为测试集，剩余未训练集
    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=0)

    readed_time = time.time()
    print("读文件时间："+str(readed_time-start))

    print("开始训练。。。")
    """
    计算判断最优划分的方法，默认是gini,对应cart算法，entropy为信息增益（对应ID3）算法
    """
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(train_features,train_labels)
    fitted_time = time.time()
    print("训练模型时间："+str(fitted_time-readed_time))

    print("预测数据。。。")
    test_predict = clf.predict(test_features)
    print("预测结果：")
    print(test_predict)
    predicted_time = time.time()
    print("预测花费时间："+str(predicted_time-fitted_time))

    score = accuracy_score(test_labels,test_predict)
    print("准确度：%f%%" % score)