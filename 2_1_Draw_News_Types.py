# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
nj_j = pd.read_excel('Temp/SelPriceDropYear_Brief_News.xlsx')
str_type = nj_j['刊期种类']
freq_type = dict.fromkeys(str_type, 0)
# 循环d 如果遇到了就在c中+1 有意思的是c中的每个键值一定在d中 所以只要遍历d 然后把value+1就可以了
for x in str_type:
    freq_type[x] += 1
fig = plt.figure()
data = pd.Series(list(dict(freq_type).values()),index=list(dict(freq_type).keys()))
print(data)
index2=['Daily','Six Times Weekly','Thrice Weekly','Five Times Weekly','Weekly','Four Times Weekly',
        'Twice Weekly','semi-monthly','Twice Ten Days']
data2 = pd.Series(list(dict(freq_type).values()),index=index2)
data2.plot.barh(color='black',alpha=1,rot=0,fontsize=10,width=0.8)
#解决中文标签字体显示问题
plt.rcParams['font.sans-serif'] = 'SimHei' # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.ylabel('Counts',fontsize=12)
# plt.title('报纸种类频次图')
plt.xlabel('Newspapers Types',fontsize=12)
fig.savefig('Figures/报纸种类频次图_无月报2.png',dpi=600,bbox_inches='tight')