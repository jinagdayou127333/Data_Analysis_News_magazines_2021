# coding=gbk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imageio import imread
import re
import jieba
import collections
nj= pd.read_excel('Temp/SelPriceDropYear_Brief_News.xlsx')
print(nj.columns)
user_labels_list = []
nj_popular=nj
# nj_popular=nj[nj['�Ƿ�������']=='��']
user_labels=nj_popular.iloc[:,1:]
for label in user_labels['��Ʒ����']:    #��Ʒ���� #������� #�������� #�ѱ���ÿһ�д浽�б�user_labels_list��
    user_labels_list.append(label)
cleanedList = [x for x in user_labels_list if str(x) != 'nan']   #ȥ����ֵ
mytext = ''.join(cleanedList)    #���б����ַ���
stopwords=['�ʺ�','����','����','�����ʺ�','�����ʺ�','���','�հ�','���ݼ��','����','���߶���',
           'CIP���ֺ�','����','����','��','ʡ','��','��','��','����','ͻ��',
           '��','��','��','��','����','Ϊ','��ֽ','��־','�ڿ�','��','����','�ṩ','��','��',
           '�й�','��','�ҹ�','Ϊ��','��Ҫ','����','��','����',
           '�й�','����','ֱϽ��','������','��','��','��','���ɹ�','���ֹ���','�ؼ�']
punctuation = """�����������磥��������������������.���������ۣܣݣޣߣ��������?????��������, ( ) -�������������� '\n' ��������?????����????�C����'?����??��?�n"""
re_punctuation = "[{}]+".format(punctuation)
mytext = re.sub(re_punctuation, "", mytext)
seg_list_exact = jieba.cut(mytext, cut_all = False) # ��ȷģʽ�ִ�
object_list = []
for word in seg_list_exact: # ѭ������ÿ���ִ�
    if word not in stopwords: # �������ȥ���ʿ���
        if word=='����' or word =='����' or word=='����' or word=='����':
            word='����'
        if word=='����' or word=='ָ��' or word=='����':
            word='����'
        if word=='��' or word=='����':
            word='��'
        if word=='�ֻ�' or word =='�ͻ���':
            word='�ͻ���'
        if word=='�˽�' or word=='�˽̰�':
            word='�˽̰�'
        if word == '�Ƽ�' or word == '��ѧ����':
            word = '�Ƽ�'
        if word == '��' or word == '����':
            word = '�ϰ���'
        object_list.append(word) # �ִ�׷�ӵ��б�
word_counts = collections.Counter(object_list)
mk = imread('wordle-word-cloud.tif')   #��ȡ���Ʊ�����״ͼƬ
wordcloud = WordCloud(font_path = "�����ֺ��μ���.ttf",background_color="white",
                      collocations=False,max_words=2000,stopwords = stopwords,mask=mk)
wordcloud.generate_from_frequencies(word_counts)
wordcloud.to_file('Figures/���ݲ�Ʒ����ı�ֽ����.png') #����ͼƬ

counts = word_counts.most_common(15)
print(dict(counts)) #��ʾ��Ƶ
fig = plt.figure()
data = pd.Series(list(dict(counts).values()),index=list(dict(counts).keys()))
data.plot.bar(color='black',alpha=1,rot=80,fontsize=10)
#������ı�ǩ������ʾ����
plt.rcParams['font.sans-serif'] = 'SimHei' # ָ��Ĭ������
plt.rcParams['axes.unicode_minus'] = False # �������ͼ���Ǹ���'-'��ʾΪ���������
plt.title('���ݲ�Ʒ����ı�ֽǰ15��ʻ�')
plt.ylabel('Ƶ��',fontsize=12)
plt.xlabel('�ʻ�',fontsize=12)
fig.savefig('Figures/���ݲ�Ʒ����ı�ֽǰ15�����.png',dpi=300,bbox_inches='tight')