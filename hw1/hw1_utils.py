## 读取数据 #
#训练集的大小是(4320,27)，因为前三列分别是”日期“，”测站“和”测项“，实际大小为(4320,24)。  
#其中，4320又为(12,20,18)，分别对应”月份“，”日期“和”测项“。

import pandas as pd
import numpy as np

test_option_index= []

def load_data():
    filepath_train = 'datas/train.csv' # 训练集的路径
    filepath_test = 'datas/test.csv'   # 测试集的路径

    data_train_orig = pd.read_csv(filepath_train,encoding='big5') # 读取训练集数据，台湾是用的big5码
    data_test_orig = pd.read_csv(filepath_test,encoding='big5',
                            header=None,
                            index_col=0) # 读取测试集数据，没有列表，但是有index

    test_options = data_train_orig.iloc[0:18,2].values                # 所有测项名称
    test_option_index = {t:i for t,i in zip(test_options,range(18))}  # 测项的索引字典

    data_train_orig = data_train_orig.iloc[:,3:] # 去掉前三列无用数据
    data_train_orig[data_train_orig == 'NR'] = 0 # 设置雨量为NR的数据为0
    data_train_orig = data_train_orig.to_numpy()

    data_test_orig = data_test_orig.iloc[:,1:] # 去掉标识'测项'的第一列
    data_test_orig[data_test_orig == 'NR'] = 0
    data_test_numpy = data_test_orig.to_numpy()

    # 转化每月数据 #
    #把数据(12\*20\*18,24)转化成(12,18,24\*18)的数据格式，对应(月份，测项，小时*每月天数)

    data_train_month = np.empty([12,18,24*20])  # 定义初始值

    for month in range(12):
        for day in range(20):
            row_start = (month * 20 + day) * 18 # 起始行，每月20天，每天18个特征
            row_end = row_start + 18            # 每天18个特征
            data_day = data_train_orig[row_start:row_end,:]        # 获得数据切片
            
            col_start = day * 24                # 每天有24小时
            col_end = col_start + 24
            data_train_month[month,:,col_start:col_end] = data_day # 设置数据切片

    # 生成样本数据 #
    #因为是预测第十天的数据，所以连续九天的数据合并成一条数据,X_train_full的维度(12\*471,18\*9)

    sample_number_month = 24 * 20 - 10 + 1                     # 计算每月的样本数

    X_train_full = np.empty([sample_number_month * 12,18 * 9]) # 初始化训练集X
    y_train_full = np.empty([sample_number_month * 12,])       # 初始化训练集y

    for month in range(12):
        for sample in range(sample_number_month):
            col_start = sample           # 开始列
            col_end   = col_start + 9    # 取9列数据
            data_X = data_train_month[month,:,col_start:col_end].reshape(1,-1) # 变成一行数据
            data_y = data_train_month[month,test_option_index['PM2.5'],col_end] # y是下一列数据
            
            row = month * sample_number_month + sample # 插入数据的行号
            X_train_full[row,:] = data_X
            y_train_full[row] = data_y

    test_number = data_test_numpy.shape[0] // 18
    X_test = np.empty([test_number,18 * 9])
    for sample in range(test_number):
        row_start = sample * 18
        row_end = row_start + 18
        X_test[sample] = data_test_numpy[row_start:row_end,:].reshape(1,-1)
        
    X_train_pd = pd.DataFrame(X_train_full)
    X_test_pd  = pd.DataFrame(X_test)
    # y_train_pd = pd.DataFrame(y_train_full)

    return X_train_pd,y_train_full,X_test_pd,test_option_index,test_options

## 定义选择测项的Transformer ##
from sklearn.base import BaseEstimator,TransformerMixin
class SelectTestOptionTransformer(BaseEstimator,TransformerMixin):
    
    def __init__(self,test_option_index,exclude=[],exclude_index=None):
        self.test_option_index = test_option_index
        self.exclude = exclude
        self.exclude_index = exclude_index
        
    def fit(self,X,y=None):
        return self

    def transform(self,X,y =None):
        exclude = []

        if self.exclude_index:
            exclude = [self.exclude_index]

        elif self.exclude:
            exclude = [self.test_option_index[e] for e in self.exclude] # 获得测项对应的索引值
        
        X_copy = X.copy()                                   
        
        for e in exclude:
            X_copy.drop(columns=[9 * e + i for i in range(9)],inplace=True,errors='ignore')                            
        return X_copy

## 定义画学习曲线函数 ##
import matplotlib.pyplot as plt
def plot_validation_curve(train_scores,valid_scores,param_range,ylim=(0.7,1)):
    plt.plot(param_range,train_scores.mean(axis=1),label='train scores') # 画训练集的分数
    
    valid_mean = valid_scores.mean(axis=1)
    plt.plot(param_range,valid_mean,label='valid scores')                # 画验证集的分数

    valid_max = valid_mean.max()                           # 验证集中的最大分数
    valid_argmax = param_range[valid_mean.argmax()]        # 验证集最大分数对应的参数取值
    plt.plot(valid_argmax,valid_max,'x',)                  # 画出最大参数的位置
    plt.text(valid_argmax,valid_max + 0.01,
             '({},{:.4f})'.format(valid_argmax,valid_max)) # 画出最大参数的坐标值

    plt.legend()
    plt.ylim(ylim)
    plt.show()

    return valid_argmax,valid_max # 返回最大得分和对应的参数值


from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
class SearchUtils() :

    def __init__(self,model,X,y):
        self.model = model
        self.X = X
        self.y = y
        
    ## 定义搜索函数 ##
    def search(self,param,param_range,ylim=(0.7,1),xlabel=None):
        train_scores,valid_scores = validation_curve(self.model,
                                                     self.X,
                                                     self.y,
                                                     param,
                                                     param_range,
                                                     verbose=1,
                                                     n_jobs=4,
                                                     error_score=0)
        if not xlabel:
            xlabel = param_range
        param_max,score_max = plot_validation_curve(train_scores,valid_scores,xlabel,ylim=ylim)
        return (param_max,score_max,train_scores,valid_scores)

    ## 定义方格搜索函数 ##
    def grid_search(self,gridparam):
        grid_search = GridSearchCV(self.model,gridparam,verbose=1,n_jobs=4)

        history = grid_search.fit(self.X,self.y)

        print(history.best_params_)
        print(history.best_score_)
        
        return history

