# -*- coding: UTF-8 -*-
#!usr/bin/python3
#Copyright 2020 Think Team, Inc. All right reserved.
#Author: James_Bobo
#Completion Date: 2020-10-29
# kalman filter
import numpy as np
from scipy.optimize import curve_fit
from FuzzyControl import FuzzyCtr


class KalFuzzy:
    def __init__(self, parameter_a, 
                 x_minSOC_range_a, 
                 x_E_range_a, 
                 y_alpha_range_a, 
                 parameter_b, 
                 x_minSOC_range_b,
                 x_E_range_b, 
                 y_alpha_range_b):
        self.A = FuzzyCtr(parameter=parameter_a, 
                          x_minSOC_range=x_minSOC_range_a, 
                          x_E_range=x_E_range_a,
                          y_alpha_range=y_alpha_range_a)
        self.A.rule()
        self.B = FuzzyCtr(parameter=parameter_b, 
                          x_minSOC_range=x_minSOC_range_b, 
                          x_E_range=x_E_range_b,
                          y_alpha_range=y_alpha_range_b)
        self.B.rule()
        self.new_SOH = None

    def kf_fl(self, minSOC, maxSOC, Time, SOH, flag=0):
        # 每个数据是一个列向量
        x1 = minSOC  # 起始SOC             ********
        x2 = maxSOC  # 终止SOC             ********
        n = Time.shape[0]  # 计算连续n个时刻    ********
        Q = 0.03 ** 2
        P = np.zeros(n)
        P[0] = 1  # 初始是否有误差？

        # 初始化
        z = SOH
        self.new_SOH = np.zeros(n)
        xhatminus = np.zeros(n)  # % SOH的先验估计。即在k-1时刻，对k时刻SOH做出的估计
        Pminus = np.zeros(n)  # % 先验估计的方差
        K = np.zeros(n)  # % 卡尔曼增益。
        alpha = np.zeros(n)  # % SOH权重

        if flag == 0:
            self.new_SOH[0] = np.max(SOH)  # % 理论上是取平均值，由于原始估计不理想，故在此取最大值  ****
            if self.new_SOH[0] >= 100:
                self.new_SOH[0] = 99
        else:
            self.new_SOH[0] = np.mean(SOH)

        # 循环执行，找到新的SOH
        for k in range(1, n):
            # 时间更新（预测）
            xhatminus[k] = self.new_SOH[k - 1]  # 用上一时刻的最优估计值来作为对当前时刻的SOH的预测
            Pminus[k] = P[k - 1] + Q  # 预测的方差为上一时刻SOH最优估计值的方差与过程方差（为常数）之和
            # 测量更新（校正）
            e1 = abs(z[k] - self.new_SOH[k - 1]) / self.new_SOH[k - 1]

            # 卡尔曼滤波更新
            if x1[k] > 50 and e1 > 0.15:
                alpha[k] = 100
            elif x2[k] >= 99:
                alpha[k] = self.A.defuzzy_a(x1[k], e1)
            else:
                alpha[k] = self.B.defuzzy_a(x1[k], e1)
            R = (alpha[k] * 1.5) ** 2
            K[k] = Pminus[k] / (Pminus[k] + R)  # 计算卡尔曼增益
            self.new_SOH[k] = xhatminus[k] + K[k] * (
                    z[k] - xhatminus[k])  # 结合当前时SOH计算值，对上一时刻的预测进行校正，得到校正后的最优估计。该估计具有最小均方差
            P[k] = (1 - K[k]) * Pminus[k]  # 计算最终估计值的方差
            print("k",k)
            print("self.new_SOH",self.new_SOH)


        return self.new_SOH


# 阿伦尼乌斯拟合
def arrhenius_func(x, a, b, c):
    ahenius = a * (x ** b) + c
    return ahenius

 # 阿伦尼乌斯拟合
class AlFit:   
    def __init__(self):
        self.__a = 0
        self.__b = 0
        self.__c = 0
        self.isFitted = 0

    def arrhenius(self, x):
        return self.__a * (x ** self.__b) + self.__c

    def fit(self, x_data, y_data):
        try:
            popt, pcov = curve_fit(arrhenius_func, x_data, y_data)
            self.__a = popt[0]
            self.__b = popt[1]
            self.__c = popt[2]
            print("popt",popt)
            print("pcov",pcov)
            print("a:", self.__a, ',b:', self.__b, ',c:', self.__c)
            return 1
        except ValueError:
            print("error")
            return -1

    def get_para(self):
        return self.__a, self.__b, self.__c

    def set_para(self, a, b, c):
        self.__a = a
        self.__b = b
        self.__c = c
