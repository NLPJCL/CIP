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
        self.len_tags = 0
        self.len_feature = 0
        self.dic_tags = {}  # 词性的字典
        self.weight = []  # 特征向量的值
        self.v = []

        self.pos_dev = []
        self.words_dev = []

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

    def readdata(self):
        self.words, self.pos=self.readfile('train.conll')
        self.words_dev, self.pos_dev = self.readfile('dev.conll')

    def create_feature_templates_global(self, words, index, tag, pre_tag):
        f = []
        # words_of_sentence
        wos = words
        if index == 0:
            wi_minus = '$$'
            ci_minus_minus1 = '$'
            pre_tag = '<BOS>'
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
        f.append('01:' + str(tag) + '*' + str(pre_tag))
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
            f.append('14:'+ str(tag) +'*'+wos[index][0:i+1])
            f.append('15:'+ str(tag) +'*'+wos[index][-(i+1)::])
        return f

    def get_max_score_path(self, sentence, WA= False):
        #WA 权重累加
        path_max = []
        if WA:
            w=self.v
        else:
            w=self.weight
        dim_tags = self.len_tags  # 词性维度
        len_sen = len(sentence)  # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        for i in range(dim_tags):
            f = self.create_feature_templates_global(sentence, 0,self.tags[i],'<BOS>')
            h = [self.feature[j] for j in f if j in self.feature]
            score_prob[0,i]=np.sum(w[h])
        if len_sen > 1:
            for i in range(1, len_sen):
                for j in range(dim_tags):
                    temp_prob=np.zeros(dim_tags)
                    for k in range(dim_tags):  # 路径搜索
                        f=self.create_feature_templates_global(sentence,i,self.tags[j],self.tags[k])
                        h=[self.feature[fi] for fi in f if fi in self.feature]
                        temp_prob[k]=np.sum(w[h])+score_prob[i - 1][k]
                    score_prob[i,j]=np.max(temp_prob)
                    score_path[i,j]=np.argmax(temp_prob)
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
                    pretag = 'NULL'
                else:
                    pretag = self.pos[index_sen][index_word - 1]
                f = self.create_feature_templates_global(sen, index_word, tag, pretag)
                for i in f:
                    if i not in self.feature:
                        self.feature[i] = len(self.feature)
                if tag not in self.tags:
                    self.tags.append(tag)
        self.dic_tags = {value: i for i, value in enumerate(self.tags)}
        self.len_feature = len(self.feature)
        self.len_tags = len(self.tags)
        self.weight = np.zeros( self.len_feature, dtype=np.int)
        self.v = np.zeros_like(self.weight)
        print "特征向量数目：" + str(self.len_feature)
        print "词性数目：" + str(self.len_tags)

    def perceptron_online_training(self, iteration=20):
        for it in range(0, iteration):
            start_time=datetime.today()
            for index_sen in range(0, len(self.words)):
                sen = self.words[index_sen]
                max_tag = self.get_max_score_path(sen)
                right_tag = self.pos[index_sen]
                for i in range(0, len(max_tag)):
                    tag_m = max_tag[i]
                    tag_p = right_tag[i]
                    if tag_m != tag_p:
                        if i == 0:
                            pretag_m = 'NULL'
                            pretag_p = 'NULL'
                        else:
                            pretag_m = max_tag[i - 1]
                            pretag_p = right_tag[i - 1]
                        f_m = self.create_feature_templates_global(sen, i, tag_m, pretag_m)
                        f_p = self.create_feature_templates_global(sen, i, tag_p, pretag_p)
                        fm_index=[self.feature[fi] for fi in f_m if fi in self.feature]
                        fp_index = [self.feature[fi] for fi in f_p if fi in self.feature]
                        self.weight[fm_index] -= 1
                        self.weight[fp_index] += 1
            self.testdata('train')
            self.testdata('dev')
            with open('result_b.txt', 'a') as fr:
                fr.write('迭代：'+str(it) +'\t'+'用时'+str(datetime.today()-start_time)+'s'+'\n')

    def output(self):
        with open('model.txt', 'w') as fw:
            for i in self.feature:
                index = self.feature[i]
                value = self.weight[index]
                fw.write(i.encode('UTF-8')+'\t' + str(value) + '\n')

    def test_sentence(self, sentence, right_tag):
        match = 0
        total =len(sentence)
        max_tag = self.get_max_score_path(sentence,WA=False)
        for i,j in zip(max_tag,right_tag):
            if i == j:
                match += 1
        return match, total

    def testdata(self,dataset):
        match = 0
        total = 0
        if dataset=='train':
            words=self.words
            pos=self.pos
        elif dataset=='dev':
            words=self.words_dev
            pos=self.pos_dev
        for i in range(0,len(words)):
            m,t=self.test_sentence(words[i],pos[i])
            match += m
            total += t
        accuracy = match * 1.0 / total
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open('result_b.txt', 'a') as fr:
            fr.write(dataset + 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy


if __name__ == '__main__':
    glm = GlobalLinearModel()
    glm.readdata()
    glm.create_feature_space()
    glm.perceptron_online_training()
    glm.output()
