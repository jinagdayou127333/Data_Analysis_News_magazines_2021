# coding=gbk
#�����ڿ���Ʒ����Ĵʴ�
import pandas as pd
import matplotlib.pyplot as plt
import re
import jieba
plt.style.use('seaborn-white')
punctuation = """�����������磥��������������������.���������ۣܣݣޣߣ��������?????��������, 
    ( ) -�������������� '\n' ��������?????����????�C����'?����??��?�n"""
re_punctuation = "[{}]+".format(punctuation)
nj = pd.read_excel('Temp/Final_Brief_Journal.xlsx')
nj['�������'] = nj['�������'].astype(str)
user_labels_list = []
user_labels = nj.reset_index(drop = True)
print(user_labels.info())
stopwords=['�ʺ�','����','����','�����ʺ�','�����ʺ�','���','�հ�','���ݼ��','����','���߶���','CIP���ֺ�','����','����','��','ʡ','��','��',
           '��','��','��','��','����','Ϊ','��ֽ','��־','�ڿ�','��','����','�ṩ','��','��','����','�й�','��','�ҹ�','Ϊ��','��Ҫ','����','��','����',
           '�й�','����','ֱϽ��','������','��','��','��','���ɹ�','���ֹ���','�ؼ�']
words = []
for i, row in user_labels.iterrows():
    word = jieba.cut(re.sub(re_punctuation, "", row['�������']),cut_all = False) # test=0.2->0.9881
    if word not in stopwords:
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
        result = ' '.join(word)
        words.append(result)
# print(words)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(words)
X = X.toarray()
# print(X)
words_bag = vect.vocabulary_
print('��������{}'.format(len(vect.vocabulary_)))
print(words_bag) #��ʾ�ʴ�
words_bag2 = vect.get_feature_names()
df_x = pd.DataFrame(X)
df_cplb = pd.DataFrame(X,columns=words_bag2)
df_cplb.to_csv('Temp/Introduction_Words_Journal.csv', encoding='gbk')