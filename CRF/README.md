## 全局线性模型Global-Linear-Model

### 一、目录文件

```
./data/:
    train.conll: 训练集
    dev.conll: 开发集
./big-data/
    train.conll: 训练集
    dev.conll: 开发集
    test.conll: 测试集
./result:
    origin_W.txt: 初始版本，小数据测试，使用W作为权重的评价结果
    origin_V.txt: 初始版本，小数据测试，使用V作为权重的评价结果
    partial_feature_W: 使用部分特征优化后，小数据测试，使用W作为权重的结果
    partial_feature_V: 使用部分特征优化后，小数据测试，使用V作为权重的结果
./src:
    global-linear-model.py: 初始版本的代码
    global-linear-model-partial-feature.py: 优化后的代码,速度还是过慢
    global-linear-model-partial-feature-2D.py: 进一步用numpy优化后的代码
    config.py: 配置文件，用字典存储每个参数
./README.md: 使用说明
```



### 二、运行

##### 1.运行环境

​    python 3.6.3

##### 2.运行方法

```python
#配置文件中各个参数
config = {
    'train_data_file': './data/train.conll',   #训练集文件,大数据改为'./big-data/train.conll'
    'dev_data_file': './data/dev.conll',       #开发集文件,大数据改为'./big-data/dev.conll'
    'test_data_file': './data/dev.conll',      #测试集文件,大数据改为'./big-data/test.conll'
    'averaged': False,                         #是否使用averaged percetron
    'iterator': 20,                            #最大迭代次数
    'shuffle': False                           #每次迭代是否打乱数据
}
```

```bash
$ cd ./Global-Linear-Model
$ python src/global-linear-model.py                    #修改config.py文件中的参数
$ python src/global-linear-model-partial-feature-2D.py    #修改config.py文件中的参数
```

##### 3.参考结果

##### (1)小数据测试

训练集：data/train.conll

开发集：data/dev.conll

| 文件         | global-linear-model.py | global-linear-model.py | Global-linear-model-partial-feature-2D.py | Global-linear-model-partial-feature-2D.py |
| :----------- | ------------ | ------------ | --------------- | --------------- |
| 特征权重     | W            | V            | W               | V               |
| 是否打乱数据 | 否 | 否 | 否 | 否 |
| 执行时间     | 13,519s | 13,698s | 1,219s | 1,221ss |
| 训练集准确率 | 99.95% | 99.47% | 99.91% | 99.70% |
| 开发集准确率 | 86.65% | 87.48% | 87.32% | 88.07% |
| 迭代次数     | 17 | 14 | 15 | 17 |
| 最大迭代次数 | 20           | 20           | 20 | 20 |

注：用numpy二维矩阵整体操作可以大大加快viterbi的速度。

##### (2)大数据测试

训练集：big-data/train.conll

开发集：big-data/dev.conll

测试集：big-data/test.conll

| 文件         | global-linear-model.py | global-linear-model.py | Global-linear-model-partial-feature-2D.py | Global-linear-model-partial-feature-2D.py |
| :----------- | ---------------------- | ---------------------- | ----------------------------------------- | ----------------------------------------- |
| 特征权重     | W                      | V                      | W                                         | V                                         |
| 是否打乱数据 | 否                     | 否                     | 否                                        | 否                                        |
| 执行时间     |                        |                        |                                           |                                           |
| 训练集准确率 |                        |                        |                                           |                                           |
| 开发集准确率 |                        |                        |                                           |                                           |
| 测试集准确率 |                        |                        |                                           |                                           |
| 迭代次数     |                        |                        |                                           |                                           |
| 最大迭代次数 | 20                     | 20                     | 20                                        | 20                                        |
