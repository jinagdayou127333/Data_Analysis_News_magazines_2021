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

nj = pd.read_excel('Temp/Final_Brief_Comb_Journal.xlsx')
print(nj.info())
nj_product = nj.loc[:,['���д���','����','���','����','��Ʒ����','������','�Ƿ�������']]
lst_tar=np.zeros([nj_product.shape[0],1])
for id in range(nj_product.shape[0]):
    if nj_product.iloc[id,:]['�Ƿ�������']=='��':
        lst_tar[id]= 1
    else:
        lst_tar[id]= 0
nj_product['�������']=lst_tar
# print(nj_product['�������'])
nj_product_pop = nj_product.loc[:,['����','���','���д���','����','��Ʒ����','������','�������']]
nj_product_corr=nj_product_pop.corr()
print(nj_product_corr)#���ϵ��