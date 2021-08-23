# -*- coding: UTF-8 -*-
#!usr/bin/python3
#Copyright 2020 Think Team, Inc. All right reserved.
#Author: James_Bobo
#Completion Date: 2020-10-29
from SOHCal import *
from matplotlib import pyplot as plt

def estimate_soh(data):
    parameter_a = { 'x_min_zmf10': 0, 
                    'x_min_zmf11': 8.333, 
                    'x_min_trimf1': [0, 16.67, 33.33],
                    'x_min_trimf2': [8.333, 25, 41.67], 
                    'x_min_trimf3': [16.67, 33.33, 50], 
                    'x_min_smf10': 41.67,
                    'x_min_smf11': 50, 
                    'x_E_zmf10': 0, 
                    'x_E_zmf11': 0.1125, 
                    'x_E_smf10': 0.1125, 
                    'x_E_smf11': 0.15,
                    'y_zmf10': 0.01, 
                    'y_zmf11': 0.1575, 
                    'y_trimf1': [0.01, 0.1575, 0.305],
                    'y_trimf2': [0.1575, 0.305, 0.4525],
                    'y_trimf3': [0.305, 0.4525, 0.6], 
                    'y_smf10': 0.4525, 
                    'y_smf11': 0.6, 
                    'y_zmf20': 0.01, 
                    'y_zmf21': 0.1,
                    'y_trimf4': [0.541, 5.91, 11.81]}
    x_minSOC_range_a = np.arange(0, 50, 0.01, np.float32)#0-0.5等差排列的5000个数
    x_E_range_a = np.arange(0, 0.15, 0.001, np.float32)#0-0.15等差排列的150个数
    y_alpha_range_a = np.arange(0.01, 0.6, 0.001, np.float32)#0-0.6等差排列的590个数

    parameter_b = { 'x_min_zmf10': 0, 
                    'x_min_zmf11': 8.333, 
                    'x_min_trimf1': [0, 16.67, 33.33],
                    'x_min_trimf2': [8.333, 25, 41.67], 
                    'x_min_trimf3': [16.67, 33.33, 50], 
                    'x_min_smf10': 41.67,
                    'x_min_smf11': 50, 
                    'x_E_zmf10': 0, 
                    'x_E_zmf11': 0.1125, 
                    'x_E_smf10': 0.1125, 
                    'x_E_smf11': 0.15,
                    'y_zmf10': 0.5, 
                    'y_zmf11': 0.625, 
                    'y_trimf1': [0.5, 0.625, 0.75],
                    'y_trimf2': [0.625, 0.75, 0.875],
                    'y_trimf3': [0.75, 0.875, 1], 
                    'y_smf10': 0.875, 
                    'y_smf11': 1, 
                    'y_zmf20': 0.5, 
                    'y_zmf21': 0.575,
                    'y_trimf4': [0.95, 5.5, 10.5]}
    x_minSOC_range_b = np.arange(0, 50, 0.1, np.float32)#0-0.5等差排列的5000个数
    x_E_range_b = np.arange(0, 0.15, 0.001, np.float32)#0-0.15等差排列的150个数
    y_alpha_range_b = np.arange(0.5, 1, 0.001, np.float32)#0.6-1等差排列的590个数

    KF = KalFuzzy(parameter_a=parameter_a, 
                  x_minSOC_range_a=x_minSOC_range_a, 
                  x_E_range_a=x_E_range_a,
                  y_alpha_range_a=y_alpha_range_a, 
                  parameter_b=parameter_b, 
                  x_E_range_b=x_E_range_b,
                  x_minSOC_range_b=x_minSOC_range_b, 
                    y_alpha_range_b=y_alpha_range_b)
    print("kf",KF)

    data_cal = DataCalculation()
    data_cal.get_singlecar_soh(car=data, InfoFrom=0, df=0, para60=60, capCase=0, constCap=0)
    result_T, SOH = karman(KF, data_cal.abstable)
    return result_T, SOH


if __name__ == '__main__':    
    data_input = pd.read_csv(r'./153Ah充电数据.csv', encoding="gbk")
    print(data_input.head())
    result_T, SOH = estimate_soh(data=data_input)
    print("SOH",SOH)
    print("result_T",result_T)
    print("len(SOH)",len(SOH),"len(result_T)",len(result_T))
    plt.plot(result_T, SOH)
    plt.show()
