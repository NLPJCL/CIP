#!\usr\bin\python
# coding=UTF-8
import numpy as np
from datetime import *


class GlobalLinearModel:
    def __init__(self):
        self.pos = []  # words对应的词性
        self.words = []  # 句子分割成词的词组列表的列表
        self.feature = {}  # 特征向量的字典
        self.tags = []  # 词性的集合 n
        self.len_tags=0
        self.len_feature=0
        self.dic_tags={} # 词性的字典
        self.weight = []  # 特征向量的值
        self.v=[]  # 权重累加
        self.pos_dev=[]
        self.words_dev=[]
        self.words_test = []
        self.pos_test = []
        self.output_file='test_c1_bigdata_average.txt'
        self.average=True

    def readfile(self,filename):
        with open(filename, 'r') as ft:
            words=[]
            pos=[]
            temp_words = []
            temp_poss = []
            for line in ft:
                if len(line) > 1:
                    temp_word = line.strip().split('\t')[1].decode('utf-8')
                    temp_pos = line.strip().split('\t')[3]
                    if temp_pos in self.tags:
                        self.tags.append(temp_pos)
                    temp_words.append(temp_word)
                    temp_poss.append(temp_pos)
                else:
                    words.append(temp_words)
                    pos.append(temp_poss)
                    temp_words = []
                    temp_poss = []
            return words,pos

    def getdata(self):
        self.words, self.pos = self.readfile('../train.conll')
        self.words_dev, self.pos_dev = self.readfile('../dev.conll')
        self.words_test, self.pos_test = self.readfile('../test.conll')

    def create_feature_templates_part(self, words, index):
        f = []
        # words_of_sentence
        wos = words
        if index == 0:
            wi_minus = '$$'
            ci_minus_minus1 = '$'
        else:
            wi_minus = wos[index - 1]
            ci_minus_minus1 = wi_minus[-1]

        if index == len(wos) - 1:
            wi_plus = '##'
            ci_plus0 = '#'
        else:
            wi_plus = wos[index + 1]
            ci_plus0 = wi_plus[0]
        ci0 = wos[index][0]
        ci_minus1 = wos[index][-1]
        f.append('02:' + wos[index])
        f.append('03:' + wi_minus)
        f.append('04:' + wi_plus)
        f.append('05:' + wos[index] + '*' + ci_minus_minus1)
        f.append('06:' + wos[index] + '*' + ci_plus0)
        f.append('07:' + ci0)
        f.append('08:' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:' + wos[index][i])
            f.append('10:' + wos[index][0] + '*' + wos[index][i])
            f.append('11:' + wos[index][-1] + '*' + wos[index][i])
        if len(wos[index]) == 1:
            f.append('12:' + wos[index] + '*' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(0, len(wos[index]) - 1):
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        for i in range(0, 4):
            if i > len(words[index]) - 1:
                break
            f.append('14:' + wos[index][0:i + 1])
            f.append('15:' + wos[index][-(i + 1)::])
        return f

    def create_feature_templates_head(self, pre_tag):
        return '01:' + str(pre_tag)

    def create_feature_templates(self, words, index,pre_tag):
        if index==0:
            pre_tag='<BOS>'
        f=self.create_feature_templates_part(words,index)
        f.append(self.create_feature_templates_head(pre_tag))
        return f

    def get_max_score_path(self, sentence, average= False):
        #WA 权重累加
        path_max = []
        if average:
            w=self.v
        else:
            w=self.weight
        dim_tags = self.len_tags  # 词性维度
        len_sen = len(sentence)  # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        f = self.create_feature_templates(sentence, 0, '<BOS>')
        h = [self.feature[j] for j in f if j in self.feature]
        score_prob[0]=np.sum(w[:, h],axis=1)
        h=[self.feature[self.create_feature_templates_head(self.tags[i])] for i in range(dim_tags)]
        prob_head=w[:,h]
        if len_sen > 1:
            for i in range(1, len_sen):
                f = self.create_feature_templates_part(sentence, i)
                h = [self.feature[ii] for ii in f if ii in self.feature]
                sum_up = np.sum(w[:, h],axis=1)
                for j in range(dim_tags):
                    temp_prob=prob_head[j]+score_prob[i-1]
                    m=np.argmax(temp_prob)
                    score_path[i, j] = m
                    score_prob[i, j] = temp_prob[m] + sum_up[j]
        # 回溯最优路径
        a = score_prob[len_sen - 1]
        max_point = np.argmax(a)
        for i in range(len_sen - 1, 0, -1):
            path_max.append(self.tags[max_point])
            max_point = score_path[i][max_point]
        path_max.append(self.tags[max_point])
        path_max.reverse()
        return path_max

    def create_feature_space(self):
        for index_sen in range(0, len(self.words)):
            sen = self.words[index_sen]
            for index_word in range(0, len(sen)):
                tag = self.pos[index_sen][index_word]
                if index_word == 0:
                    pretag = '<BOS>'
                else:
                    pretag = self.pos[index_sen][index_word - 1]
                f = self.create_feature_templates(sen, index_word, pretag)
                for i in f:
                    if i not in self.feature:
                        self.feature[i] = len(self.feature)
                if tag not in self.tags:
                    self.tags.append(tag)
        self.dic_tags={value:i for i,value in enumerate(self.tags)}
        self.len_feature=len(self.feature)
        self.len_tags=len(self.tags)
        self.weight = np.zeros((self.len_tags,self.len_feature),dtype=np.int)
        self.v = np.zeros((self.len_tags,self.len_feature ))
        print "特征向量数目：" + str(self.len_feature)
        print "词性数目：" + str(self.len_tags)

    def perceptron_online_training(self, iteration=40,average=False):
        R=np.zeros_like(self.weight)
        k=1
        for it in range(0, iteration):
            start_time = datetime.today()
            print "迭代："+str(it)
            for index_sen in range(0, len(self.words)):
                sen = self.words[index_sen]
                max_tag = self.get_max_score_path(sen)
                right_tag = self.pos[index_sen]
                for i in range(0, len(max_tag)):
                    if max_tag[i]!=right_tag[i]:
                        index_tag_m=self.dic_tags[max_tag[i]]
                        index_tag_p=self.dic_tags[right_tag[i]]
                        if i == 0:
                            pretag_m = '<BOS>'
                            pretag_p = '<BOS>'
                        else:
                            pretag_m = max_tag[i - 1]
                            pretag_p = right_tag[i - 1]
                        f_m = self.create_feature_templates(sen, i, pretag_m)
                        f_p = self.create_feature_templates(sen, i, pretag_p)
                        index_f_m=[self.feature[i] for i in f_m if i in self.feature]
                        index_f_p = [self.feature[i] for i in f_p if i in self.feature]
                        if average:
                            t_m= k * np.ones_like(index_f_m) - R[index_tag_m,index_f_m]
                            R[index_tag_m,index_f_m] = k
                            t_p= k * np.ones_like(index_f_p) - R[index_tag_p, index_f_p]
                            R[index_tag_p, index_f_p] = k
                            self.v[index_tag_m,index_f_m] += t_m * self.weight[index_tag_m,index_f_m]
                            self.v[index_tag_p, index_f_p] += t_p * self.weight[index_tag_p, index_f_p]
                            k += 1
                        self.weight[index_tag_m, index_f_m] -= 1
                        self.weight[index_tag_p, index_f_p] += 1
            if average:
                self.v += (k * np.ones_like(self.weight) - R) * self.weight
                R = k * np.ones_like(self.v)
            self.test('test.conll')
            self.test('dev.conll')
            over_time=datetime.today()
            with open(self.output_file, 'a') as fr:
                fr.write('迭代：'+str(it) +'\t'+'用时'+str(over_time-start_time)+'s'+'\n')

    def output(self):
        with open('model.txt', 'w') as fw:
            for j in range(0,self.len_tags):
                for i in range(self.len_feature):
                    f = self.feature[i]
                    value = self.weight[j,i]
                    if value!=0:
                        fw.write(str(self.tags[j])+ '*'+f.encode('UTF-8')+'\t' + str(value)+'\n')

    def test_sentence(self, sentence, right_tag, average=False):
        match = 0
        total =len(sentence)
        max_tag = self.get_max_score_path(sentence,average)
        for i,j in zip(max_tag,right_tag):
            if i == j:
                match += 1
        return match, total

    def test(self,filename):
        match = 0
        total = 0
        if filename == 'train.conll':
            words = self.words
            pos = self.pos
        elif filename == 'test.conll':
            words = self.words_test
            pos = self.pos_test
        else:
            words = self.words_dev
            pos = self.pos_dev
        for i in range(0,len(words)):
            m,t=self.test_sentence(words[i],pos[i],average=self.average)
            match += m
            total += t
        accuracy = match * 1.0 / total
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open(self.output_file, 'a') as fr:
            fr.write(filename + '\tRight:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy

if __name__ == '__main__':
    glm = GlobalLinearModel()
    glm.getdata()
    glm.create_feature_space()
    glm.perceptron_online_training(average=True)
    #glm.output()
