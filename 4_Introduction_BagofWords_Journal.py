# coding=gbk
#产生期刊产品分类的词袋
import pandas as pd
import matplotlib.pyplot as plt
import re
import jieba
plt.style.use('seaborn-white')
punctuation = """！？。＂＃＄％＆＇（）＊＋－／：；.＜＝＞＠［＼］＾＿｀｛｜｝～?????、〃》，, 
    ( ) -《「」『』【】 '\n' 〔〕〖〗?????〝〞????–—‘'?“”??…?﹏"""
re_punctuation = "[{}]+".format(punctuation)
nj = pd.read_excel('Temp/Final_Brief_Journal.xlsx')
nj['报刊简介'] = nj['报刊简介'].astype(str)
user_labels_list = []
user_labels = nj.reset_index(drop = True)
print(user_labels.info())
stopwords=['适合','本刊','本报','本报适合','本刊适合','简介','空白','内容简介','内容','读者对象','CIP核字号','对象','读者','年','省','含','版',
           '以','的','和','是','对于','为','报纸','杂志','期刊','及','订阅','提供','等','与','及其','中国','报','我国','为主','主要','方面','类','报含',
           '中共','中央','直辖市','自治区','级','地','市','内蒙古','锡林郭勒','县级']
words = []
for i, row in user_labels.iterrows():
    word = jieba.cut(re.sub(re_punctuation, "", row['报刊简介']),cut_all = False) # test=0.2->0.9881
    if word not in stopwords:
        if word=='宣传' or word =='报导' or word=='传播' or word=='报道':
            word='报导'
        if word=='辅导' or word=='指导' or word=='培养':
            word='培养'
        if word=='新' or word=='最新':
            word='新'
        if word=='手机' or word =='客户端':
            word='客户端'
        if word=='人教' or word=='人教版':
            word='人教版'
        if word == '科技' or word == '科学技术':
            word = '科技'
        result = ' '.join(word)
        words.append(result)
# print(words)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(words)
X = X.toarray()
# print(X)
words_bag = vect.vocabulary_
print('单词数：{}'.format(len(vect.vocabulary_)))
print(words_bag) #显示词袋
words_bag2 = vect.get_feature_names()
df_x = pd.DataFrame(X)
df_cplb = pd.DataFrame(X,columns=words_bag2)
df_cplb.to_csv('Temp/Introduction_Words_Journal.csv', encoding='gbk')