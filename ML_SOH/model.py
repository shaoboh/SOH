# -*- coding:utf-8 -*-
import time
import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,BaggingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

class Model:
    def Data_filter(self,file_path, Tmin=20, SOCmax=95, SOCmin=20, dSOC_filter=30,km_new_filter=10000, rest_time_filter=0, abnormal_str=[np.inf, -np.inf,'#NAME?'], encoding='gbk'):
        '''
        对提取的充电统计数据进行筛选，筛选用于新电池SOE建模的数据
        :param file_path: 充电统计的储存路径
        :param Tmin: 充电起始最小温度筛选条件
        :param SOCmax: 充电结束SOC筛选条件
        :param SOCmin: 充电起始SOC筛选条件
        :param dSOC_filter: SOC变化筛选条件
        :param km_new_filter: 认为是新电池阶段的最大里程
        :param rest_time_filter: 可以用于SOC校正的小电流持续时间
        :param abnormal_str: 异常值处理
        :param encoding: 默认编码
        :return:None
        '''

        df=pd.read_csv(file_path,encoding=encoding)

        #异常值处理
        df=df.loc[(df.SOC丢失==0)&(df.充电起始Tmin>Tmin)&(df.充电起始SOC<SOCmin)&(df.充电结束SOC>SOCmax)&(df.SOC变化>dSOC_filter)&(df.运行里程<km_new_filter)&(df.充电前小于2A时间>= rest_time_filter),:]
        '''对df去重重复值，函数默认根据所有重复值进行了判断去重，并保留了第一行'''
        df=df.drop_duplicates()
        print('全量数据特征维度{}'.format(df.shape))
        df.replace(abnormal_str, np.nan, inplace=True)
        df=df.dropna(axis=0, how='any').reset_index(drop=True)
        print('删除异常数据特征维度{}'.format(df.shape))
        df.to_csv('模型训练数据.csv',encoding='gbk')
        return df
    def Model_train(self,df,column_name):
        '''
        对建模数据标准化后，试用几个常见的回归方法进行建模，根据前期经验以下几种方法精度较好，LinearRegression，Ridge，lasso，RandomForestRegressor，AdaBoostRegressor，GradientBoostingRegressor，BaggingRegressor
        :param column_name:用于建模的列名， list形式
        :return:None
        '''
        #1,划分测试集和训练集
        model_df=df.loc[:,column_name]
        X=model_df.values[:,:-1]
        print('变量维度{}'.format(X.shape))
        y=model_df.values[:,-1:]

        x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.25)
        print('训练数据特征维度{},标签维度{}'.format(x_train.shape,y_train.shape))
        minmax=MinMaxScaler(feature_range=(0,1))
        x_train=minmax.fit_transform(x_train)
        x_test=minmax.transform(x_test)
        joblib.dump(minmax, 'minmax.pkl')

        #2,模型选择
        # 1)LinearRegression
        time1=time.perf_counter()
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('LinearRegression, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('LinearRegression')
        plt.show()
        time2=time.perf_counter()
        print('LinearRegression,训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))
        # 2)Ridge
        time1=time.perf_counter()
        model=RidgeCV(alphas=np.array([0.001, 0.01, 0.1, 0.0001]))
        model.fit(x_train,y_train)
        print(model.alpha_)
        y_fit = model.predict(x_train)
        y_predict = model.predict(x_test)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('ridge,训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('Ridge')
        plt.show()
        joblib.dump(model, 'Ridge.pkl')
        time2=time.perf_counter()
        print('ridge,训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))

        #3)lasso
        time1=time.perf_counter()
        model = LassoCV(alphas=np.array([0.1,0.01,0.001,0.0001]))
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test).reshape(-1,1)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('lasso, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('lasso')
        plt.show()
        time2=time.perf_counter()
        print('lasso, 学习率{}, 训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.alpha_,(time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))

        # 4)RandomForestRegressor
        time1=time.perf_counter()
        model=RandomForestRegressor(max_depth=20,random_state=0,n_estimators=20)
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test).reshape(-1,1)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('RandomForestRegressor, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('RandomForestRegressor')
        plt.show()
        time2=time.perf_counter()
        print('RandomForestRegressor, 训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))
        #5)AdaBoostRegressor
        time1=time.perf_counter()
        model=AdaBoostRegressor()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test).reshape(-1,1)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('AdaBoostRegressor,训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('AdaBoostRegressor.png')
        plt.show()
        time2=time.perf_counter()
        print('AdaBoostRegressor, 训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))
        # 6)GradientBoostingRegressor
        time1=time.perf_counter()
        model=GradientBoostingRegressor()
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test).reshape(-1,1)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('GradientBoostingRegressor,训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('GradientBoostingRegressor.png')
        plt.show()
        time2=time.perf_counter()
        print('GradientBoostingRegressor, 训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))
        # 7)BaggingRegressor
        time1=time.perf_counter()
        model=BaggingRegressor(AdaBoostRegressor())
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test).reshape(-1,1)
        plt.scatter(np.arange((y_predict/y_test).shape[0]),y_predict/y_test)
        plt.title('BaggingRegressor, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format(model.score(x_train,y_train),model.score(x_test,y_test)))
        plt.savefig('BaggingRegressor.png')
        plt.show()
        time2=time.perf_counter()
        print('BaggingRegressor, 训练时间{:.4f}, 训练集精度:{:.4f}, 测试集精度:{:.4f}'.format((time2-time1),model.score(x_train,y_train),model.score(x_test,y_test)))
        return '模型训练完成'
if __name__ == '__main__':
    model=Model()
    file_path=r'F:\运行数据\A-标准化程序\SOH_model\soh_model\test\蔚来充电特征数据.csv'
    #选择用于建模的列
    model_column=['运行里程', '充电前Vmin', '充电前Vmax', '充电前Vd', '充电结束Vmin', '充电结束Vmax', '充电结束Vd',
                                                  '充电起始Tmin','充电起始Tmax', '充电结束Tmin', '充电结束Tmax', '充电起始SOC','充电结束SOC','SOC变化','充入容量']
    print(model.Model_train(model.Data_filter(file_path),model_column))
