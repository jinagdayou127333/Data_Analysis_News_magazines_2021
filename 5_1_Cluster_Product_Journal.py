# coding=gbk
import warnings
from time import time
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
cplb_qk = pd.read_csv('Temp/Product_Words_Journal.csv',encoding='GB18030')
nj = pd.read_excel('Temp/Final_Brief_Journal.xlsx')
cplb_qk=cplb_qk.iloc[:,1:]
nj=nj.iloc[:,1:]
cplb_bz_list = cplb_qk.reset_index(drop = True)
# print(cplb_bz_list.head())
tsne = TSNE(n_components=3,init='pca',random_state=0)
lb_proj = tsne.fit_transform(cplb_bz_list)
kclus = 5
kmeans = KMeans(n_clusters=kclus)
kmeans.fit(lb_proj)
clusters = kmeans.labels_.tolist()
nj['产品聚类']=clusters
nj = pd.DataFrame(nj).reset_index(drop = True)
nj.to_excel('Temp/Final_ProductCluster_Journal.xls')
