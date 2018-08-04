#!\usr\bin\python
# coding=UTF-8
import numpy as np
from datetime import *
from scipy.misc import logsumexp


class ConditionRandomField:
    def __init__(self):
        self.feature = {}  # 特征向量字典
        self.dic_pos = {}  # 词性字典
        self.weight = []  # 权重字典
        self.pos = []  # 词性列表
        self.len_feature = 0  # 特征向量维度
        self.len_pos = 0  # 词性维度
        self.sentences_train = []  # 句子（词语列表）的列表——训练集
        self.tags_train = []  # 各句词性的列表——训练集
        self.len_sentences_train = 0  # 句子数量——训练集
        self.sentences_dev = []  # 句子（词语列表）的列表——测试集
        self.tags_dev = []  # 各句词性的列表——测试集
        self.len_sentences_dev = 0  # 句子数量——测试集


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

    def create_feature_templates(self, sentence, index, pretag):
        f = []
        # words_of_sentence
        wos = sentence
        if index == 0:
            wi_minus = '$$'
            ci_minus_minus1 = '$'
            pretag = '<BOS>'
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
        f.append('01:' + str(pretag))
        f.append('02:' + wos[index])
        f.append('03:' + wi_minus)
        f.append('04:' + wi_plus)
        f.append('05:' + wos[index] + '*' + ci_minus_minus1)
        f.append('06:' + wos[index] + ci_plus0)
        f.append('07:' + ci0)
        f.append('08:' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:' + wos[index][i])
            f.append('10:' + wos[index][0] + '*' + wos[index][i])
            f.append('11:' + wos[index][-1] + '*' + wos[index][i])
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        if len(wos[index]) == 1:
            f.append('12:' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(1, 5):
            if i > len(wos[index]):
                break
            f.append('14:' + wos[index][0:i + 1])
            f.append('15:' + wos[index][-i - 1::])
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
                    pretag = 'NULL'
                else:
                    pretag = self.tags_train[i][j - 1]
                f = self.create_feature_templates(sentence, j, pretag)
                for item in f:
                    if item not in self.feature:
                        self.feature[item]=len(self.feature)
        self.len_feature = len(self.feature)
        self.weight = np.zeros((self.len_pos, self.len_feature))

    def get_score(self, sentence, index, index_pre_tag, index_tag):
            if index==0:
                pre_tag = 'NULL'
            else:
                pre_tag = self.pos[index_pre_tag]
            f = self.create_feature_templates(sentence, index, pre_tag)
            index_f=[self.feature[i] for i in f if i in self.feature]
            return np.sum(self.weight[index_tag, index_f])


    def get_score_M(self, sentence):
        # 返回得分矩阵score（句子中词语下标，前一个词词性，该词词性）
        ls=len(sentence)
        score=np.zeros((ls,self.len_pos,self.len_pos))
        f = self.create_feature_templates(sentence, 0, 'NULL')
        index_f = [self.feature[i] for i in f if i in self.feature]
        for index_tag in range(self.len_pos):
            score[0, 0, index_tag] = np.sum(self.weight[index_tag, index_f])
        for index in range(1,ls):
            f = self.create_feature_templates(sentence, index, self.pos[0])
            index_f = [self.feature[i] for i in f if i in self.feature]
            for index_pre_tag in range(self.len_pos):
                pre_tag = self.pos[index_pre_tag]
                index_f[0]=self.feature['01:' + str(pre_tag)]
                for index_tag in range(self.len_pos):
                    score[index, index_pre_tag, index_tag] = np.sum(self.weight[index_tag, index_f])
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
        return p

    def cal_grad(self, sentence,tags):
        score=self.get_score_M(sentence)
        p = self.calc_p(sentence,score)
        g = np.zeros_like(self.weight)
        for index_word in range(len(sentence)):
            if index_word==0:
                pretag='NULL'
            else:
                pretag=tags[index_word-1]
            f = self.create_feature_templates(sentence, index_word, pretag)
            tag=tags[index_word]
            index_tag = self.dic_pos[tag]
            index_f = [self.feature[i] for i in f]
            g[index_tag,index_f]+=1
        f = self.create_feature_templates(sentence, 0, 'NULL')
        fs = [self.feature[k] for k in f]
        #for t in range(self.len_pos):
        g[:, fs] -=p[0, 0, :,np.newaxis]
        for i in range(1,len(sentence)):
            f = self.create_feature_templates(sentence, i, self.pos[0])
            fs = [self.feature[k] for k in f]
            for t_ in range(self.len_pos):
                fs[0]=self.feature['01:' + str(self.pos[t_])]
                #f = self.create_feature_templates(sentence, i, self.pos[t_])
                #fs = [self.feature[k] for k in f]
                #for t in range(self.len_pos):
                    #p1 = p[i, t_, t]
                    #g[t, fs] -= p1
                g[:, fs] -=p[i,t_,:,np.newaxis]
        return g

    def training(self, iteration=40):
        for it in range(iteration):
            starttime = datetime.today()
            for index_sen in range(0, self.len_sentences_train):
                sentence = self.sentences_train[index_sen]
                tags = self.tags_train[index_sen]
                g = self.cal_grad(sentence,tags)
                self.weight += g
            print 'Time(Train):' + str((datetime.today() - starttime))
            #self.testdata('train')
            #print 'Time(Train+test train):' + str((datetime.today() - starttime))
            if it>=20:
                self.testdata('dev')
                print 'Time(all):' + str((datetime.today() - starttime))

    def get_max_score_path(self, sentence):
        MIN = -10
        path_max = []
        dim_tags = self.len_pos  # 词性维度
        len_sen = len(sentence)  # 句子单词个数
        score_prob = np.zeros((len_sen, dim_tags))  # 得分矩阵
        score_path = np.zeros((len_sen, dim_tags), dtype=np.int)  # 路径矩阵
        score_path[0] = np.array([-1] * dim_tags)
        f = self.create_feature_templates(sentence, 0, '<BOS>')
        h = [self.feature[j] for j in f if j in self.feature]
        score_prob[0]=[np.sum(self.weight[i, h]) for i in range(dim_tags)]
        score_prob[1:,:]=MIN
        if len_sen > 1:
            for i in range(1, len_sen):
                for j in range(dim_tags):
                    temp_prob = np.zeros(dim_tags)
                    f = self.create_feature_templates(sentence, i, self.pos[0])
                    h = [self.feature[ii] for ii in f if ii in self.feature]
                    sum=np.sum(self.weight[j, h[1:]])
                    for k in range(dim_tags):  # 路径搜索
                        h[0]=self.feature['01:' + str(self.pos[k])]
                        #temp_prob[k]=np.sum(self.weight[j, h])+score_prob[i - 1][k]
                        temp_prob[k] = sum+self.weight[j,h[0]] + score_prob[i - 1][k]
                    score_prob[i,j]=np.max(temp_prob)
                    score_path[i,j]=np.argmax(temp_prob)
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
        # for i,j in max_tag,right_tag:
        for i,j in zip(max_tag,right_tag):
            if i == j:
                match += 1
        return match, total

    def testdata(self, dataset):
        match = 0
        total = 0
        if dataset == 'train':
            words = self.sentences_train
            pos = self.tags_train
        elif dataset == 'dev':
            words = self.sentences_dev
            pos = self.tags_dev
        for i in range(len(words)):
            m, t = self.test_sentence(words[i], pos[i])
            match += m
            total += t
        accuracy = match * 1.0 / total
        print 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(accuracy)
        with open('result2.txt', 'a') as fr:
            fr.write(dataset + 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy


if __name__ == '__main__':
    CRF = ConditionRandomField()
    CRF.read_data()
    CRF.create_feature_space()
    CRF.training()
