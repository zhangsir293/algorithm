import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadDataSet(filename):
    with open(filename) as f:
        numFeat = len((f.readline().split('\t')))
        dataMat = []#数据矩阵
        labelMat = []#标签矩阵
        for line in f.readlines():
            lineArr = []
            curlLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curlLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curlLine[-1]))
        return dataMat,labelMat

if __name__ == '__main__':
    dataArr,classLabels = loadDataSet('horseColicTraining2.txt')#训练数据集
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')#测试数据集
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=10)
    bdt.fit(dataArr,classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr),1)))
    print("训练集分类结果：(1没病，-1有病)")
    print(predictions)
    print('训练集的错误率：%.3f%%' % float(errArr[predictions!=classLabels].sum()/len(dataArr)*100))
    predictions=bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr),1)))
    print("测试集分类结果：(1没病，-1有病)")
    print(predictions)
    print('测试集的错误率：%.3f%%' % float(errArr[predictions!=testLabelArr].sum()/len(testArr)*100))