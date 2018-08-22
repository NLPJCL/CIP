#!/usr/bin/python
# coding=UTF-8
import numpy as np
from datetime import *
from scipy.misc import logsumexp
from collections import defaultdict


class ConditionRandomField:
    def __init__(self):
        self.feature = {}  # 特征向量字典
        self.dic_pos = {}  # 词性字典
        self.weight = []  # 权重字典
        self.v=[] #  权重累加
        self.pos = []  # 词性列表
        self.len_feature = 0  # 特征向量维度
        self.len_pos = 0  # 词性维度
        self.sentences_train = []  # 句子（词语列表）的列表——训练集
        self.tags_train = []  # 各句词性的列表——训练集
        self.len_sentences_train = 0  # 句子数量——训练集
        self.sentences_dev = []  # 句子（词语列表）的列表——测试集
        self.tags_dev = []  # 各句词性的列表——测试集
        self.len_sentences_dev = 0  # 句子数量——测试集
        self.sentences_test=[]
        self.tags_test=[]   #  测试集
        self.output_file='CRF_8.txt'

        self.step_opt = False
        self.regularization = False
        self.iteration = 50

    def readfile(self, filename):
        sentences = []
        tags = []
        count = 0
        with open(filename, 'r') as fr:
            temp_words = []
            temp_tags = []
            for line in fr:
                if len(line) > 1:
                    cur_word = line.strip().split()[1].decode('utf-8')
                    cur_tag = line.strip().split()[3]
                    temp_tags.append(cur_tag)
                    temp_words.append(cur_word)
                else:
                    sentences.append(temp_words)
                    tags.append(temp_tags)
                    temp_words = []
                    temp_tags = []
                    count += 1
        return sentences, tags, count

    def read_data(self):
        self.sentences_train, self.tags_train, self.len_sentences_train = self.readfile('train.conll')
        self.sentences_dev, self.tags_dev, self.len_sentences_dev = self.readfile('dev.conll')

    def create_feature_templates(self,sentence, index,pre_tag):
        f=self.create_feature_templates_part(sentence,index)
        f.append(self.create_feature_templates_head(pre_tag))
        return f

    def create_feature_templates_head(self, pre_tag):
        return '01:' + pre_tag

    def create_feature_templates_part(self, sentence, index):
        f = []
        # words_of_sentence
        wos = sentence
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
            if i > len(wos[index]) - 1:
                break
            f.append('14:' + wos[index][0:i + 1])
            f.append('15:' + wos[index][-(i + 1)::])
        return f

    def create_feature_space(self):
        for tags in self.tags_train:
            for tag in tags:
                if tag not in self.pos:
                    self.pos.append(tag)
        self.dic_pos = {pos: index for index, pos in enumerate(self.pos)}
        self.len_pos = len(self.pos)
        for i in range(0, self.len_sentences_train):
            sentence = self.sentences_train[i]
            for j in range(0, len(self.sentences_train[i])):
                if j == 0:
                    pretag = '<BOS>'
                else:
                    pretag = self.tags_train[i][j - 1]
                f = self.create_feature_templates(sentence, j,pretag)
                for item in f:
                    if item not in self.feature:
                        self.feature[item]=len(self.feature)
        self.len_feature = len(self.feature)
        self.weight = np.zeros((self.len_pos, self.len_feature))
        self.v=np.zeros_like(self.weight)
        print "特征向量数目：" + str(self.len_feature)
        print "词性数目：" + str(self.len_pos)

    def get_score(self, sentence, index, index_pre_tag, index_tag):
            if index==0:
                pre_tag = '<BOS>'
            else:
                pre_tag = self.pos[index_pre_tag]
            f = self.create_feature_templates(sentence, index,pre_tag)
            index_f=[self.feature[i] for i in f if i in self.feature]
            return np.sum(self.weight[index_tag, index_f])

    def get_score_M(self, sentence):
        # 返回得分矩阵score（句子中词语下标，前一个词词性，该词词性）
        ls=len(sentence)
        score=np.zeros((ls,self.len_pos,self.len_pos))
        f = self.create_feature_templates_part(sentence, 0)
        f.append((self.create_feature_templates_head('<BOS>')))
        index_f = [self.feature[i] for i in f if i in self.feature]
        for index_tag in range(self.len_pos):
            score[0, 0, index_tag] = np.sum(self.weight[index_tag, index_f])
        f_head=[self.create_feature_templates_head(t) for t in self.pos]
        index_f_head=[self.feature[t] for t in f_head]
        for index in range(1,ls):
            f = self.create_feature_templates_part(sentence, index)
            index_f = [self.feature[i] for i in f if i in self.feature]
            # sum_up=np.sum(self.weight[:,index_f],axis=1)
            # score[index]=sum_up+self.weight[:,index_f_head]
            for index_tag in range(self.len_pos):
                sumup=np.sum(self.weight[index_tag, index_f])
            #     # for index_pre_tag in range(self.len_pos):
            #     #     index_f0=index_f_head[index_pre_tag]
            #     #     score[index, index_pre_tag, index_tag] = sumup+self.weight[index_tag,index_f0]
                score[index,:,index_tag]=sumup+self.weight[index_tag,index_f_head]
        return score

    def log_alpha(self, sentence,score):
        x = len(sentence)
        y = self.len_pos
        alpha = np.zeros((x, y))
        alpha[0] = score[0,0,:]
        #  改法2
        for k in range(1, x):
            alpha[k] = [logsumexp(score[k,:,t]+alpha[k-1]) for t in range(y)]
        return alpha

    def log_beta(self, sentence,score):
        x = len(sentence)
        y = self.len_pos
        beta = np.zeros((x, y))
        #  改法2
        for k in range(x-2, -1,-1):
            beta[k] = [logsumexp(score[k+1,t]+beta[k+1]) for t in range(y)]
        return beta


    def calc_p(self, sentence, score):
        alpha = self.log_alpha(sentence, score)
        beta = self.log_beta(sentence, score)

        len_sen = len(sentence)
        z = logsumexp(alpha[-1])
        p = np.ones((len_sen, self.len_pos, self.len_pos))*float("-inf")
        p[0,0,:] = beta[0] + score[0,0,:]-z
        p[1:len_sen] = score[1:len_sen] + np.expand_dims(beta[1:len_sen], axis=1) + np.expand_dims(alpha[0:len_sen - 1], axis=2) - z
        p = np.exp(p)
        print [np.sum(p[i]) for i in range(len_sen)]
        return p

    def cal_grad(self, sentence,tags):
        score=self.get_score_M(sentence)
        p = self.calc_p(sentence,score)
        g=defaultdict(float)
        f = self.create_feature_templates(sentence, 0,'<BOS>')
        fs = [self.feature[k] for k in f]
        index_tag = self.dic_pos[tags[0]]
        for i in fs:
            g[i,]-=p[0, 0, :]
            g[index_tag, i] += 1
        f_head = [self.create_feature_templates_head(t) for t in self.pos]
        index_f_head = [self.feature[t] for t in f_head]
        for i in range(1,len(sentence)):
            f = self.create_feature_templates_part(sentence, i)
            fs = [self.feature[k] for k in f]
            index_tag = self.dic_pos[tags[i]]
            f_pre = self.create_feature_templates_head(tags[i - 1])
            f_pre_index = self.feature[f_pre]
            g[index_tag, f_pre_index] += 1
            for j in fs:
                g[j,] -= np.sum(p[i],axis=0)

                g[index_tag, j] += 1

            for t_ in range(self.len_pos):
                g[index_f_head[t_],]-=p[i,t_]
        return g

    def training(self, iteration=100,step_opt=False,regulization=False):

        C = 0.0001
        global_step = 1
        decay_rate = 0.96
        decay_steps =self.len_sentences_train
        eta = 0.5
        learn_rate = eta

        init_eta=0.5
        c=0.0001
        _lamda=c/10000
        _t0=1/(_lamda*init_eta)
        _t=0

        for it in range(iteration):
            starttime = datetime.today()
            print '迭代：'+ str(it)
            for index_sen in range(0, self.len_sentences_train):
                sentence = self.sentences_train[index_sen]
                tags = self.tags_train[index_sen]
                g = self.cal_grad(sentence,tags)

                #正则+步长优化  1：
                # if regulization:
                #     self.weight *= (1 - C * eta)
                # if step_opt:
                #     self.weight+=learn_rate*g
                #     learn_rate=eta*decay_rate**(global_step/decay_steps)
                #     global_step+=1
                # else:
                #     self.weight += g

                #正则+步长优化  2：
                # if regulization:
                #     self.weight *= (1 - C)
                # if step_opt:
                #     _eta=1/(_lamda*(_t0+_t))
                #     self.weight+=_eta*g
                #     _t+=1
                # else:
                #     for k,value in g.items():
                #         if len(k)==1:
                #             # print np.shape(value)
                #             # print np.shape(self.weight[:,k])
                #             self.weight[:,k[0]]+=eta*value
                #         else:
                #             self.weight[k]+=eta*value
                    #self.weight += g

                # 正则+步长优化  3：
                if regulization:
                    self.weight *= (1 - C * learn_rate)
                if step_opt:
                    for k, value in g.items():
                        if len(k) == 1:
                            self.weight[:, k[0]] += eta * value
                        else:
                            self.weight[k] += eta * value
                    learn_rate=eta*decay_rate**(global_step/decay_steps)
                    global_step+=1
                else:
                    for k, value in g.items():
                        if len(k) == 1:
                            self.weight[:, k[0]] += eta * value
                        else:
                            self.weight[k] += eta * value

            print 'Time(Train):' + str((datetime.today() - starttime))
            self.test('dev.conll')
            with open(self.output_file, 'a') as fr:
                fr.write('迭代：'+ str(it)+'\tTime(all):' + str((datetime.today() - starttime))+ '\n')

    def get_max_score_path(self, sentence):
        path_max = []
        w=self.weight
        dim_tags = self.len_pos  # 词性维度
        len_sen = len(sentence)  # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        f = self.create_feature_templates(sentence, 0, '<BOS>')
        h = [self.feature[j] for j in f if j in self.feature]
        score_prob[0] = np.sum(w[:, h], axis=1)
        h = [self.feature[self.create_feature_templates_head(self.pos[i])] for i in range(dim_tags)]
        prob_head = w[:, h]
        if len_sen > 1:
            for i in range(1, len_sen):
                f = self.create_feature_templates_part(sentence, i)
                h = [self.feature[ii] for ii in f if ii in self.feature]
                sum_up = np.sum(w[:, h], axis=1)
                for j in range(dim_tags):
                    temp_prob = prob_head[j] + score_prob[i - 1]
                    m = np.argmax(temp_prob)
                    score_path[i, j] = m
                    score_prob[i, j] = temp_prob[m] + sum_up[j]
        # 回溯最优路径
        a = score_prob[len_sen - 1]
        max_point = np.argmax(a)
        for i in range(len_sen - 1, 0, -1):
            path_max.append(self.pos[max_point])
            max_point = score_path[i][max_point]
        path_max.append(self.pos[max_point])
        path_max.reverse()
        return path_max

    def test_sentence(self, sentence, right_tag):
        match = 0
        total = len(sentence)
        max_tag = self.get_max_score_path(sentence)
        for i,j in zip(max_tag,right_tag):
            if i == j:
                match += 1
        return match, total

    def test(self,filename):
        match = 0
        total = 0
        if filename == 'train.conll':
            words = self.sentences_train
            pos = self.pos
        elif filename == 'test.conll':
            words = self.sentences_test
            pos = self.tags_test
        else:
            words = self.sentences_dev
            pos = self.tags_dev
        for i in range(0,len(words)):
            m,t=self.test_sentence(words[i],pos[i])
            match += m
            total += t
        accuracy = match * 1.0 / total
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open(self.output_file, 'a') as fr:
            fr.write(filename + '\tRight:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy

if __name__ == '__main__':
    CRF = ConditionRandomField()
    CRF.read_data()
    CRF.create_feature_space()
    CRF.training(iteration=CRF.iteration,step_opt=CRF.step_opt,regulization=CRF.regularization)
