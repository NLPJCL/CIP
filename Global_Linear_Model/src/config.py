
config = {
    'train_data_file': '../data/train.conll',    #训练集文件
    'dev_data_file': '../data/dev.conll',        #开发集文件
    'test_data_file': 'None',       #测试集文件
    'averaged': False,                          #是否使用averaged percetron
    'iterator': 100,                             #最大迭代次数
    'stop_iterator': 10,                        #迭代stop_iterator次性能没有提升则结束
}
