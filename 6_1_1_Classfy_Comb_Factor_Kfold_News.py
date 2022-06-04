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

nj = pd.read_excel('Temp/Final_Brief_Comb_News.xlsx')
print(nj.info())
nj_product = nj.loc[:,['���','���д���','����','��Ʒ����','�Ƿ�������']] #add related features
# nj_product = nj.loc[:,['����','���','���д���','����','��Ʒ����','������','�Ƿ�������']] # add all features
lst_tar=np.zeros([nj_product.shape[0],1])
for id in range(nj_product.shape[0]):
    if nj_product.iloc[id,:]['�Ƿ�������']=='��':
        lst_tar[id]= 1
    else:
        lst_tar[id]= 0
nj_product['�������']=lst_tar

Data_=nj_product.iloc[:,[0,1,2,3,4]]
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


# ## ������
# clf = DecisionTreeClassifier(max_depth=16)
# clf.fit(NX,Ny)
# # ������������
# print('\n����������:', clf.feature_importances_, '\n')

# ## ֧��������
# #clf = Pipeline([("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1,loss="hinge")),])
# clf = Pipeline([("poly_features",PolynomialFeatures(degree=3)),("scaler",StandardScaler()),("linear_svc",LinearSVC(C=10,loss="hinge")),])
# clf.fit(NX,Ny)


## ������
# clf = MLPClassifier()
# clf.fit(NX,Ny)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(NX,Ny)

scores = cross_val_score(clf, NX, Ny, cv=5)  #cvΪ����������
print(scores)  # ��ӡ���ÿ�ε����Ķ���ֵ��׼ȷ�ȣ�
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # ��ȡ�������䡣��Ҳ���Ǿ�ֵ�ͷ��