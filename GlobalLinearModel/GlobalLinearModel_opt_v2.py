#!\usr\bin\python
# coding=UTF-8
import numpy as np
import time


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
        self.v=[]

        self.pos_dev=[]
        self.words_dev=[]

        self.start_time = time.time()

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

    def create_feature_templates_global(self, words, index, pre_tag):
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
        f.append('01:' + str(pre_tag))
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
        for i in range(0,len(wos[index])):
            if i>=4:
                break
            f.append('14:'+'*'+wos[index][0:i+1])
            f.append('15:'+'*'+wos[index][-(i+1)::])
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
        f = self.create_feature_templates_global(sentence, 0, '<BOS>')
        h = [self.feature[j] for j in f if j in self.feature]
        score_prob[0]=[np.sum(w[i, h]) for i in range(dim_tags)]
        if len_sen > 1:
            for i in range(1, len_sen):
                for j in range(dim_tags):
                    temp_prob = np.zeros(dim_tags)
                    f = self.create_feature_templates_global(sentence, i, self.tags[0])
                    h = [self.feature[ii] for ii in f if ii in self.feature]
                    sumup=np.sum(w[j, h[1:]])
                    for k in range(dim_tags):  # 路径搜索
                        h=self.feature['01:' + str(self.tags[k])]
                        #temp_prob[k]=np.sum(self.weight[j, h])+score_prob[i - 1][k]
                        #temp_prob[k] = sumup+w[j,h[0]] + score_prob[i - 1][k]
                        temp_prob[k]=w[j,h]+score_prob[i - 1][k]
                    score_prob[i,j]=np.max(temp_prob)+sumup
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
                f = self.create_feature_templates_global(sen, index_word, pretag)
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

    def perceptron_online_training(self, iteration=40):
        for it in range(0, iteration):
            self.start_time = time.time()
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
                            pretag_m = 'NULL'
                            pretag_p = 'NULL'
                        else:
                            pretag_m = max_tag[i - 1]
                            pretag_p = right_tag[i - 1]
                        f_m = self.create_feature_templates_global(sen, i, pretag_m)
                        f_p = self.create_feature_templates_global(sen, i, pretag_p)
                        index_f_m=[self.feature[i] for i in f_m if i in self.feature]
                        self.weight[index_tag_m,index_f_m] -= 1
                        index_f_p = [self.feature[i] for i in f_p if i in self.feature]
                        self.weight[index_tag_p, index_f_p] += 1
                        self.v += self.weight
            self.testdata('train')
            self.testdata('dev')
            over_time=time.time()
            with open('result_a.txt', 'a') as fr:
                fr.write('迭代：'+str(it) +'\t'+'用时'+str(over_time-self.start_time)+'s'+'\n')


    def output(self):
        with open('model.txt', 'w') as fw:
            for j in range(0,self.len_tags):
                for i in range(self.len_feature):
                    f = self.feature[i]
                    value = self.weight[j,i]
                    if value!=0:
                        fw.write(str(self.tags[j])+  '*'+f.encode('UTF-8')+'\t' + str(value)+'\n')

    def test_sentence(self, sentence, right_tag):
        match = 0
        total =len(sentence)
        max_tag = self.get_max_score_path(sentence,WA=True)
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
        with open('result_a.txt', 'a') as fr:
            fr.write(dataset + 'Right:' + str(match) + 'Total:' + str(total) + 'Accuracy:' + str(
                accuracy) + '\n')
        return accuracy

if __name__ == '__main__':
    glm = GlobalLinearModel()
    glm.readdata()
    glm.create_feature_space()
    glm.perceptron_online_training()
    glm.output()
