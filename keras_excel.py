# -*- coding: utf-8 -*-
#!/usr/bin/python


import pandas as pd
import numpy as np
import jieba
import random
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU



#read articles
#pos=pd.read_excel('pos2.xls',header=None,index=None)
#neg=pd.read_excel('neg2.xls',header=None,index=None)
pos=pd.read_excel('pos2.xls',header=0,index=None)
neg=pd.read_excel('neg2.xls',header=0,index=None)
#print(pos)

#pos+neg
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目
print('-----------------'+'neg_len:%s'%neglen+'-----------------')
print('-----------------'+'pos_len:%s'%poslen+'-----------------')

#cut
cw = lambda x: list(jieba.cut(x)) #定义分词函数   #synom,stopwords? effects
pn['words'] = pn['article'].apply(cw)
#print(pn)
print('pn.shape=pos+neg',pn.shape)

#read articles which are need predict
comment=pd.read_excel('pre2.xls',header=0,index=None) #读入评论内容
comment = comment[comment['article'].notnull()] #仅读取非空评论
print('-----------------'+'comment_len:%s'%len(comment)+'-----------------')
comment['words'] = comment['article'].apply(cw) #评论分词

#article+comment build dict
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)
print('-----------------'+'d2v_train:'+'-----------------')
print(d2v_train)

w = [] #将所有词语整合在一起
for i in d2v_train:
	w.extend(i)

word_dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
word_dict['id']=list(range(1,len(word_dict)+1))
get_sent = lambda x: list(word_dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢


#limit word numbers
#maxlen=50 #top words
maxlen=5
print('-----------------'+'Pad sequences (samples x time)'+'-----------------')
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
print(pn)

#random articles
pn['id']=list(range(1,len(pn)+1))
pn['random_id']=random.sample(pn['id'],len(pn))
pn=pn.sort(['random_id'],ascending=True)


#train and test
train_percent=0.6
train_num=int(round((len(pn))*train_percent))
test_num=int(round((len(pn))*(1-train_percent)))

x_train= np.array(list(pn['sent']))[0:train_num] #训练集
y_train= np.array(list(pn['score']))[0:train_num]
x_test = np.array(list(pn['sent']))[0:test_num] #测试集
y_test = np.array(list(pn['score']))[0:test_num] 


#x_train= np.array(list(pn['sent']))[::2] #训练集
#y_train= np.array(list(pn['score']))[::2]
#x_test = np.array(list(pn['sent']))[1::2] #测试集
#y_test = np.array(list(pn['score']))[1::2]



xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['score']))

#model
print('-----------------'+'Build model'+'-----------------')
model = Sequential()
model.add(Embedding(len(word_dict)+1, 256))
model.add(LSTM(128)) # try using a GRU instead, for fun
#model.add(LSTM(output_dim=128, input_shape=256))
model.add(Dropout(0.5))
#model.add(Dense(128, 1))
model.add(Dense(input_dim=128,output_dim=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

#train
model.fit(x_train, y_train, batch_size=16, nb_epoch=10) 
#model.fit(x_train, y_train, batch_size=16, nb_epoch=10,validation_data=(x_test, y_test)) 

classes = model.predict_classes(x_test)
acc = np_utils.accuracy(classes, y_test)

print('predict:\n%s'%classes)
print('reality',y_test)
print('Test accuracy:', acc)


print('*'*20)
#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#print(loss_and_metrics)
hist=model.fit(x_train, y_train, validation_split=0.2)
print(hist.history)


print('-'*50)


#--------------------------------------------------------------
#predict
comment['sent'] = comment['words'].apply(get_sent)
comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen))
x_pre=np.array(list(comment['sent']))
pre=model.predict_classes(x_pre)
#print('pre:\n%s'%pre)

#write predict result to comment
comment['pre_score']=pd.DataFrame(pre)
print(comment)


#out to excel
comment_out=pd.DataFrame({'id':comment['id'],'article':comment['article'],'score':comment['pre_score']})
comment_out.to_excel('comment_out.xls','Sheet1')





