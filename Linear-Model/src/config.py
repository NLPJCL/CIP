config = {
    'train_data_file': './big-data/train.conll',    #训练集文件
    'dev_data_file': './big-data/dev.conll',        #开发集文件
    'test_data_file': './big-data/test.conll',       #测试集文件
    'averaged': False,                          #是否使用averaged percetron
    'iterator': 20,                             #最大迭代次数
    'shuffle': False                            #每次迭代是否打乱数据
}