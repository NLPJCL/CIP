#!\usr\bin\python
# coding=UTF-8
import math
import numpy as np
from scipy.misc import logsumexp
from datetime import *

class log_linear_model:
    def __init__(self):
        self.sentences = []  # 句子的列表
        self.pos = []  # words对应的词性
        self.words = []  # 句子分割成词的词组列表的列表
        self.words_dev = []
        self.pos_dev = []
        self.dic_feature = {}  # 特征向量的字典
        self.dic_tags = {}  # 词性的集合
        self.tags = []  # 词性的列表
        self.len_feature = 0
        self.len_tags = 0
        self.weight = []  # 特征权重
        self.v = []  # 权重累加

    def readfile(self, filename):
        words = []
        pos = []
        with open(filename, 'r') as ft:
            temp_words = []
            temp_poss = []
            for line in ft:
                if len(line) > 1:
                    temp_word = line.strip().split('\t')[1].decode('utf-8')
                    temp_pos = line.strip().split('\t')[3]
                    temp_words.append(temp_word)
                    temp_poss.append(temp_pos)
                else:
                    words.append(temp_words)
                    pos.append(temp_poss)
                    temp_words = []
                    temp_poss = []
        return words, pos

    def getdata(self):
        self.words, self.pos = self.readfile('train.conll')
        self.words_dev, self.pos_dev = self.readfile('dev.conll')

    def create_feature_templates(self,words,index,tag):
        f=[]
        #words_of_sentence
        wos=words
        if index==0:
            wi_minus='$$'
            ci_minus_minus1='$'
        else:
            wi_minus = wos[index - 1]
            ci_minus_minus1 = wi_minus[-1]

        if index==len(wos)-1:
            wi_plus='##'
            ci_plus0='#'
        else:
            wi_plus = wos[index + 1]
            ci_plus0 = wi_plus[0]
        ci0=wos[index][0]
        ci_minus1=wos[index][-1]
        f.append('02:' + str(tag) + '*' + wos[index])
        f.append('03:' + str(tag) + '*' + wi_minus)
        f.append('04:' + str(tag) + '*' + wi_plus)
        f.append('05:' + str(tag) + '*' + wos[index] + '*' + ci_minus_minus1)
        f.append('06:' + str(tag) + '*' + wos[index] + ci_plus0)
        f.append('07:' + str(tag) + '*' + ci0)
        f.append('08:' + str(tag) + '*' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:' + str(tag) + '*' + wos[index][i])
            f.append('10:' + str(tag) + '*' + wos[index][0] + '*' + wos[index][i])
            f.append('11:' + str(tag) + '*' + wos[index][-1] + '*' + wos[index][i])
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + str(tag) + '*' + wos[index][i] + '*' + 'consecutive')
        if len(wos[index]) == 1:
            f.append('12:' + str(tag) + '*' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(0,len(wos[index])):
            if i>=4:
                break
            f.append('14:'+str(tag)+'*'+wos[index][0:i+1])
            f.append('15:'+str(tag)+'*'+wos[index][-(i+1)::])
        return f

    def create_feature_space(self):
        for i in range(0,len(self.words)):
            cur_words=self.words[i]
            for j in range(0,len(cur_words)):
                tag=self.pos[i][j]
                f=self.create_feature_templates(cur_words,j,tag)
                for feat in f:
                    if feat not in self.dic_feature:
                        self.dic_feature[feat]=len(self.dic_feature)
                if tag not in self.tags:
                    self.tags.append(tag)
        self.len_feature=len(self.dic_feature)
        self.len_tags=len(self.tags)
        self.weight=np.zeros(self.len_feature)
        self.v=np.zeros(self.len_feature)
        self.dic_tags={value:i for i,value in enumerate(self.tags)}
        print('特征空间维度：%d'%self.len_feature)
        print ('词性维度：%d'%self.len_tags)

    def get_score(self,f):
        f_index=[self.dic_feature[fi] for fi in f if fi in self.dic_feature]
        return np.sum(self.weight[f_index])

    def get_score_average(self,f):
        f_index = [self.dic_feature[fi] for fi in f if fi in self.dic_feature]
        return np.sum(self.v[f_index])

    def get_max_tag(self,words,index):
        temp_score=np.zeros(self.len_tags)
        for t in range(self.len_tags):
            f=self.create_feature_templates(words,index,self.tags[t])
            temp_score[t]=self.get_score(f)
        index=np.argmax(temp_score)
        return self.tags[index]

    def get_prob(self,words,index):
        score=np.zeros(self.len_tags)
        for i in range(self.len_tags):
            f=self.create_feature_templates(words,index,self.tags[i])
            score[i]=self.get_score(f)
        lse=logsumexp(score)
        score-=lse
        return np.exp(score)

    def SGD_training(self,iteration=40):
        SA = True
        g=np.zeros_like(self.weight)
        B=50
        b=0
        k=0
        eta=1.0

        C = 0.0001
        global_step = 1
        decay_rate = 0.96
        decay_steps = 100000
        if SA:
            eta = 0.5
        learn_rate = eta

        for it in range(0,iteration):
            print '当前迭代次数'+str(it)
            start=datetime.today()
            for index_sen in range(0,len(self.words)):
                cur_sen=self.words[index_sen]
                for index_word in range(0,len(cur_sen)):
                    tag=self.pos[index_sen][index_word]
                    f_tag=self.create_feature_templates(cur_sen,index_word,tag)
                    index_f=[self.dic_feature[i] for i in f_tag if i in self.dic_feature]
                    g[index_f]+=1
                    prob=self.get_prob(cur_sen,index_word)
                    for t in range(self.len_tags):
                        p_tag=self.create_feature_templates(cur_sen,index_word,self.tags[t])
                        index_f = [self.dic_feature[i] for i in p_tag if i in self.dic_feature]
                        g[index_f] -= prob[t]
                    b=b+1
                    if B==b:
                        if SA:
                            self.weight*=(1-C*learn_rate)
                        self.weight+=eta*g
                        k+=1
                        b=0
                        g = np.zeros_like(self.weight)

                        if SA:
                            learn_rate = eta * decay_rate ** (global_step / decay_steps)
                            global_step += 1


            self.test('dev.conll')
            print 'Time:' + str(datetime.today() - start)
        print '模型更新次数'+str(k)

    def output(self):
        with open('model.txt','w+') as fm:
            for i in self.dic_feature:
                fm.write(i.encode('utf-8')+'\t'+str(self.dic_feature[i])+'\n')

    def test_sentence(self,words,tags):
        right=0
        sumup=len(words)
        for i in range(0,sumup):
            max_tag=self.get_max_tag(words,i)
            if max_tag==tags[i]:
                right+=1
        return right,sumup

    def test(self,filename):
        right=0
        total=0
        words=[]
        pos=[]
        if filename=='train.conll':
            words=self.words
            pos=self.pos
        else:
            words=self.words_dev
            pos=self.pos_dev
        for w,p in zip(words,pos):
            r,t=self.test_sentence(w,p)
            right+=r
            total+=t
        pricision=1.0*right/total
        print('正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision))
        with open('result_SA.txt','a+') as fr:
            fr.write(filename+ '正确：'+str(right)+'总数：'+str(total)+'正确率:'+str(pricision)+'\n')

if __name__=='__main__':
    lm = log_linear_model()
    lm.getdata()
    lm.create_feature_space()
    lm.SGD_training()
    lm.output()
