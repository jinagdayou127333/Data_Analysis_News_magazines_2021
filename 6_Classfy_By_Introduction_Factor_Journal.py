# coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

nj = pd.read_excel('Temp/Final_IntroductionCluster_Journal.xls')
print(nj.info())
nj_product = nj.loc[:,['发行次数','单价','年价','版面','简介聚类','是否畅销报刊']]
lst_tar=np.zeros([nj_product.shape[0],1])
for id in range(nj_product.shape[0]):
    if nj_product.iloc[id,:]['是否畅销报刊']=='是':
        lst_tar[id]= 1
    else:
        lst_tar[id]= 0
nj_product['畅销与否']=lst_tar
# print(nj_product['畅销与否'])
nj_product_pop = nj_product.loc[:,['单价','年价','发行次数','版面','简介聚类','畅销与否']]
nj_product_corr=nj_product_pop.corr()
print(nj_product_corr)#相关系数

ls_factor=['单价','年价','发行次数','版面','简介聚类']
Data_=nj_product_pop.iloc[:,[0,2,3,4,5]]
x_ = Data_.iloc[:, :4]
y_ = Data_.iloc[:, 4]

# Synthetic Minority Over-sampling Technique 合成少数类样本过采样技术
from imblearn.over_sampling import SMOTE

ros = SMOTE(random_state=0)
X_oversampled, y_oversampled = ros.fit_resample(x_, y_)
X = X_oversampled.values.astype(np.float64)

NX = MinMaxScaler().fit_transform(X)  # 归一化处理
encoder = OrdinalEncoder()
y_val = np.matrix(y_oversampled.values.reshape(-1, 1))
Ny = encoder.fit_transform(y_val)
X_train, X_test, Y_train, Y_test = train_test_split(NX, Ny, test_size=0.8, random_state=1)  # 拆分数据集
# 乱序
np.random.seed(12)
np.random.shuffle(X_test)
np.random.seed(12)
np.random.shuffle(Y_test)


## 决策树
clf = DecisionTreeClassifier(max_depth=16)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
# 各特征贡献率
print('\n特征贡献率:', clf.feature_importances_, '\n')

'''
## 支持向量机
#clf = Pipeline([("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1,loss="hinge")),])
clf = Pipeline([("poly_features",PolynomialFeatures(degree=3)),("scaler",StandardScaler()),("linear_svc",LinearSVC(C=10,loss="hinge")),])
cls_pl = clf.fit(X_train,Y_train)
Y_pred = cls_pl.predict(X_test)

## 简单贝叶斯
# clf = GaussianNB()
# cls_pl = clf.fit(X_train, Y_train)
# Y_pred = cls_pl.predict(X_test)

## 神经网络
clf = MLPClassifier()
clf.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
'''

# 正确率
train_score = clf.score(X_train, Y_train)
test_score = accuracy_score(Y_pred, Y_test)
print('训练准确度：\n', train_score)
print('测试准确度：\n', test_score)
print('使用的分类器是', clf, ';\n测试集上分类表现如下:')
print(classification_report(Y_test, Y_pred, target_names=['0', '1']))

# 画图
guess = ["0", "1"]
fact = ["0", "1"]
classes = list(set(fact))
classes.sort(reverse=True)
plt.figure(figsize=(12, 10))  # 设置plt窗口的大小
cm = confusion_matrix(Y_test, Y_pred)
print('混淆矩阵：\n', cm)
plt.imshow(cm, cmap=plt.cm.Blues)
indices = range(len(cm))
indices2 = range(3)
plt.xticks(indices, classes, rotation=40, fontsize=18)
plt.yticks([0.00, 1.00], classes, fontsize=18)
plt.ylim(1.5, -0.5)  # 设置y的纵坐标的上下限

plt.title("Confusion matrix", fontdict={'weight': 'normal', 'size': 18})

# 设置color bar的标签大小
cb = plt.colorbar()
cb.ax.tick_params(labelsize=24)
plt.xlabel('Predict label', fontsize=24)
plt.ylabel('True label', fontsize=24)

for first_index in range(len(cm)):
    for second_index in range(len(cm[first_index])):

        if cm[first_index][second_index] > 200:
            color = "red"
        else:
            color = "black"
        plt.text(first_index, second_index, cm[first_index][second_index],
                 fontsize=24, color=color, weight='bold', verticalalignment='center', horizontalalignment='center', )
plt.show()