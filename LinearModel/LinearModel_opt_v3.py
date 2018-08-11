#!\usr\bin\python
# coding=UTF-8
import numpy as np
from datetime import *


class linear_model:
    def __init__(self):
        self.sentences = []  # 句子的列表
        self.pos = []  # words对应的词性
        self.words = []  # 句子分割成词的词组列表的列表
        self.words_dev = []
        self.pos_dev = []
        self.words_test = []
        self.pos_test = []
        self.dic_feature = {}  # 特征向量的字典
        self.dic_tags = {}  # 词性的集合
        self.tags = []  # 词性的列表
        self.len_feature = 0
        self.len_tags = 0
        self.weight = []  # 特征权重
        self.v = []  # 权重累加
        self.average=False

    def readfile(self,filename):
        words=[]
        pos=[]
        with open(filename,'r') as ft:
            temp_words = []
            temp_poss = []
            for line in ft:
                if len(line)>1:
                    temp_word=line.strip().split('\t')[1].decode('utf-8')
                    temp_pos=line.strip().split('\t')[3]
                    temp_words.append(temp_word)
                    temp_poss.append(temp_pos)
                else:
                    words.append(temp_words)
                    pos.append(temp_poss)
                    temp_words = []
                    temp_poss = []
        return words,pos

    def getdata(self):
        self.words,self.pos=self.readfile('train.conll')
        self.words_dev,self.pos_dev=self.readfile('dev.conll')
        self.words_test, self.pos_test = self.readfile('test.conll')

    def create_feature_templates(self, words, index):
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
        f.append('04:'+ wi_plus)
        f.append('05:'+ wos[index] + '*' + ci_minus_minus1)
        f.append('06:'+ wos[index] + '*'+ ci_plus0)
        f.append('07:' + ci0)
        f.append('08:' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:'+ wos[index][i])
            f.append('10:'+ wos[index][0] + '*' + wos[index][i])
            f.append('11:'+ wos[index][-1] + '*' + wos[index][i])
            if wos[index][i] == wos[index][i - 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        if len(wos[index]) == 1:
            f.append('12:'+wos[index]+'*'+ci_minus_minus1 + '*' + ci_plus0)
        for i in range(0, len(wos[index]) - 1):
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        for i in range(0, len(wos[index])):
            if i >= 4:
                break
            f.append('14:' + wos[index][0:i + 1])
            f.append('15:' + wos[index][-(i + 1)::])
        return f

    def create_feature_space(self):
        for i in range(0, len(self.words)):
            cur_words = self.words[i]
            for j in range(0, len(cur_words)):
                tag = self.pos[i][j]
                f = self.create_feature_templates(cur_words, j)
                for feat in f:
                    if feat not in self.dic_feature:
                        self.dic_feature[feat] = len(self.dic_feature)
                if tag not in self.tags:
                    self.tags.append(tag)
        self.len_feature = len(self.dic_feature)
        self.len_tags = len(self.tags)
        self.weight = np.zeros((self.len_tags,self.len_feature))
        self.v = np.zeros((self.len_tags,self.len_feature))
        self.dic_tags = {value: i for i, value in enumerate(self.tags)}
        print('特征空间维度：%d' % self.len_feature)
        print ('词性维度：%d' % self.len_tags)

    def get_score(self, f,tag_index,average=False):
        f_index=[self.dic_feature[i] for i in f if i in self.dic_feature]
        if not average:
            score =np.sum(self.weight[tag_index,f_index])
        else:
            score = np.sum(self.v[tag_index, f_index])
        return score

    def get_max_tag(self, words, index,average=False):
        temp_score=np.zeros(self.len_tags)
        fs=[]
        for i in range(0,self.len_tags):
            f = self.create_feature_templates(words, index)
            fs.append(f)
            temp_score[i] = self.get_score(f,i,average)
        index=np.argmax(temp_score)
        return self.tags[index],fs[index]

    def online_training(self,iteration=20,average=False):
        R = np.zeros_like(self.weight)  # 时间戳标记
        k = 1  # weight 更新次数
        for it in range(0, iteration):
            print '当前迭代：' + str(it)
            start_time = datetime.today()
            for index_sen in range(0,len(self.words)):
                cur_sen=self.words[index_sen]
                for index_word in range(0,len(cur_sen)):
                    max_tag, f_max_tag = self.get_max_tag(cur_sen, index_word)
                    right_tag=self.pos[index_sen][index_word]
                    if max_tag!=right_tag:
                        f_right_tag = self.create_feature_templates(self.words[index_sen], index_word)
                        index_right_tag=self.dic_tags[right_tag]
                        index_max_tag=self.dic_tags[max_tag]
                        f_right_index = [self.dic_feature[i] for i in f_right_tag]
                        f_max_index = [self.dic_feature[i] for i in f_max_tag if i in self.dic_feature]
                        if average:
                            t_right= k * np.ones_like(f_right_index) - R[index_right_tag,f_right_index]
                            t_max= k * np.ones_like(f_max_index) - R[index_max_tag, f_max_index]
                            self.v[index_right_tag,f_right_index] += t_right * self.weight[index_right_tag,f_right_index]
                            self.v[index_max_tag,f_max_index] += t_max * self.weight[index_max_tag,f_max_index]
                            R[index_right_tag,f_right_index] = k
                            R[index_max_tag,f_max_index] = k
                            k += 1
                        self.weight[index_right_tag, f_right_index] += 1
                        self.weight[index_max_tag,f_max_index] -= 1
                        #self.v+=self.weight
            self.test('test.conll')
            self.test('dev.conll')
            with open('result_opt.txt', 'a+') as fr:
                fr.write('时长：'+str(datetime.today()-start_time)+'\n')

    def output(self):
        with open('model2.txt', 'w+') as fm:
            for i_tag in range(0,self.len_tags):
                for i in self.dic_feature:
                    myweight=self.weight[i_tag, self.dic_feature[i]]
                    if myweight!=0:
                        fm.write(i.encode('utf-8') + self.tags[i_tag] + '\t'+str(myweight) + '\n')
        print 'Output Successfully'

    def test_sentence(self, words, tags):
        right = 0
        sumup = len(words)
        for i in range(0, sumup):
            max_tag ,_s= self.get_max_tag(words, i,self.average)
            if max_tag == tags[i]:
                right += 1
        return right, sumup

    def test(self, filename):
        right = 0
        total = 0
        words = []
        pos = []
        if filename == 'train.conll':
            words = self.words
            pos = self.pos
        elif filename=='test.conll':
            words = self.words_test
            pos = self.pos_test
        else:
            words = self.words_dev
            pos = self.pos_dev
        for w, p in zip(words, pos):
            r, t = self.test_sentence(w, p)
            right += r
            total += t
        pricision = 1.0 * right / total
        print('正确：' + str(right) + '总数：' + str(total) + '正确率:' + str(pricision))
        with open('result_opt.txt', 'a+') as fr:
            fr.write(filename + '正确：' + str(right) + '总数：' + str(total) + '正确率:' + str(pricision) + '\n')

if __name__ == '__main__':
    lm = linear_model()
    lm.getdata()
    lm.create_feature_space()
    lm.online_training(40,lm.average)
    #lm.output()
