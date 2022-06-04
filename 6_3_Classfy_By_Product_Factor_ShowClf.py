# coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

nj = pd.read_excel('Temp/Final_ProductCluster_Journal.xls')
print(nj.info())
nj_product = nj.loc[:,['���д���','����','���','����','��Ʒ����','�Ƿ�������']]
lst_tar=np.zeros([nj_product.shape[0],1])
for id in range(nj_product.shape[0]):
    if nj_product.iloc[id,:]['�Ƿ�������']=='��':
        lst_tar[id]= 1
    else:
        lst_tar[id]= 0
nj_product['�������']=lst_tar
# print(nj_product['�������'])
nj_product_pop = nj_product.loc[:,['����','���','���д���','����','��Ʒ����','�������']]
nj_product_pop = nj_product_pop.rename(columns={'����':'UnitPrice','���':'YearPrice',
               '���д���':'Issues','����':'Papers','��Ʒ����':'ProductClass','�������':'BestSell'})
nj_product_corr=nj_product_pop.corr()
print(nj_product_corr)#���ϵ��

ls_factor=['����','���','���д���','����','��Ʒ����']
Data_=nj_product_pop.iloc[:,[0,2,3,4,5]]
x_ = Data_.iloc[:, :4]
y_ = Data_.iloc[:, 4]

# Synthetic Minority Over-sampling Technique �ϳ���������������������
from imblearn.over_sampling import SMOTE

ros = SMOTE(random_state=0)
X_oversampled, y_oversampled = ros.fit_resample(x_, y_)
X = X_oversampled.values.astype(np.float64)

NX = StandardScaler().fit_transform(X)  # ��һ������
# NX = MinMaxScaler().fit_transform(X)  # ��һ������
encoder = OrdinalEncoder()
y_val = np.matrix(y_oversampled.values.reshape(-1, 1))
Ny = encoder.fit_transform(y_val)
# X_train, X_test, Y_train, Y_test = train_test_split(NX, Ny, test_size=0.2, random_state=1)  # ������ݼ�
# ����
np.random.seed(12)
np.random.shuffle(NX)
np.random.seed(12)
np.random.shuffle(Ny)

## ������
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(NX,Ny)
# ������������
print('\n����������:', clf.feature_importances_, '\n')

import graphviz
from sklearn.tree import export_graphviz
dot_dat=export_graphviz(
    clf,
    out_file='journal_tree.dot',
    feature_names=x_.columns,
    rounded=True,
    filled=True
)
graph = graphviz.Source(dot_dat)#����graphvizģ���Source()�������Խ���ת��Ϊgv�ļ�
