# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_excel('data/neg.xlsx',header=None,index=None)
    pos=pd.read_excel('data/pos.xlsx',header=None,index=None)
    nan=pd.read_excel('data/nan.xlsx',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)
    nan['words'] = nan[0].apply(cw)

    #print pos['words']
    #use 1 for positive sentiment, 0 for negative,-1 for nan
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg)), -np.ones(len(nan))))
    print y
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'], nan['words'])), y, test_size=0.4, random_state=0)
    
    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train,x_test
 

#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
    
#计算词向量
'''
class gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5,min_count=5, max_vocab_size=None, sample=0.001,seed=1,workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5,cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0,trim_rule=None, sorted_vocab=1, batch_words=10000)

参数：
·  sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
·  window：表示当前词与预测词在一个句子中的最大距离是多少
·  alpha: 是学习速率
·  seed：用于随机数发生器。与初始化词向量有关。
·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
·  workers参数控制训练的并行数。
·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
·  iter： 迭代次数，默认为5
·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
·  batch_words：每一批的传递给线程的单词的数量，默认为10000
'''
def get_train_vecs(x_train,x_test):

    #Initialize model and build vocab
    n_dim = 300
    imdb_w2v = Word2Vec(size=n_dim, min_count=5)
    imdb_w2v.build_vocab(x_train)
    
    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('svm_data/train_vecs.npy',train_vecs)
    print train_vecs.shape
    #Train word2vec on test tweets
    imdb_w2v.train(x_test)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print test_vecs.shape


def get_data():
    train_vecs=np.load('svm_data/train_vecs.npy')
    y_train=np.load('svm_data/y_train.npy')
    test_vecs=np.load('svm_data/test_vecs.npy')
    y_test=np.load('svm_data/y_test.npy') 
    return train_vecs,y_train,test_vecs,y_test
    

##训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print clf.score(test_vecs,y_test)
    
    
##得到待预测单个句子的词向量    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    #print train_vecs.shape

    return train_vecs
    
####对单个句子进行情感判断    
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('svm_data/svm_model/model.pkl')
     
    result = clf.predict(words_vecs)
    return result
    
def train():
    ##导入文件，处理保存为向量
    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
    svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型

if __name__=='__main__':
    #train()
##对输入句子情感进行判断
    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(string)
    
