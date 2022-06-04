# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
nj_j = pd.read_excel('Temp/SelPriceDropYear_Brief_News.xlsx')
str_type = nj_j['��������']
freq_type = dict.fromkeys(str_type, 0)
# ѭ��d ��������˾���c��+1 ����˼����c�е�ÿ����ֵһ����d�� ����ֻҪ����d Ȼ���value+1�Ϳ�����
for x in str_type:
    freq_type[x] += 1
fig = plt.figure()
data = pd.Series(list(dict(freq_type).values()),index=list(dict(freq_type).keys()))
print(data)
index2=['Daily','Six Times Weekly','Thrice Weekly','Five Times Weekly','Weekly','Four Times Weekly',
        'Twice Weekly','semi-monthly','Twice Ten Days']
data2 = pd.Series(list(dict(freq_type).values()),index=index2)
data2.plot.barh(color='black',alpha=1,rot=0,fontsize=10,width=0.8)
#������ı�ǩ������ʾ����
plt.rcParams['font.sans-serif'] = 'SimHei' # ָ��Ĭ������
plt.rcParams['axes.unicode_minus'] = False # �������ͼ���Ǹ���'-'��ʾΪ���������
plt.ylabel('Counts',fontsize=12)
# plt.title('��ֽ����Ƶ��ͼ')
plt.xlabel('Newspapers Types',fontsize=12)
fig.savefig('Figures/��ֽ����Ƶ��ͼ_���±�2.png',dpi=600,bbox_inches='tight')