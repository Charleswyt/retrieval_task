# -*- coding: utf-8 -*-
import yaml
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import sys
sys.setrecursionlimit(1000000)

nan=pd.read_excel('data/nan.xlsx',header=None,index=None)
neg=pd.read_excel('data/neg.xlsx',header=None,index=None)
pos=pd.read_excel('data/pos.xlsx',header=None,index=None) #读取训练语料完毕
nan['mark']=2
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目
 
cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
 
comment = pd.read_excel('data/sum.xlsx') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词 
 
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
 
w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)
 
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))
 
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)  
 
maxlen = 50
 
print "Pad sequences (samples x time)" 
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
 
x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
 
print 'Build model...' 
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print 'Fit model...'  
model.fit(xa, ya, batch_size=32, nb_epoch=4) #训练时间为若干个小时
 
classes = model.predict_classes(xa)
acc = np_utils.accuracy(classes, ya)
print 'Test accuracy:', acc 
