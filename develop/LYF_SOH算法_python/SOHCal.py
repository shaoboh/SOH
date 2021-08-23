# -*- coding: UTF-8 -*-
#!usr/bin/python3
#Copyright 2020 Think Team, Inc. All right reserved.
#Author: James_Bobo
#Completion Date: 2020-10-28
import pandas as pd
from KalmanFuzzy import *

# 获得SOH平均衰减速率
def rate_of_decrease(SOH, Time):
    rate = (SOH[0] - SOH[-1]) / (Time[-1] - Time[0])
    return rate

# 卡尔曼滤波
def karman(KF, abstable):
    print("abstable",abstable)
    SOH = np.array(abstable['Soh'])
    Time = np.array(abstable['Time'])
    minSOC = np.array(abstable['minSOC'])
    maxSOC = np.array(abstable['maxSOC'])

    if SOH.shape[0] < 10:
        # 数据异常车辆
        print('too small\n')
        return []

    SOH3 = KF.kf_fl(minSOC, maxSOC, Time, SOH)

    abstable['resultSOH'] = SOH3
    abstable['Time'] = (Time - np.min(Time)) / (60 * 60 * 24)
    Time = np.array(abstable['Time'])
    print("Time+++++++++++++++++++++++++++++++++++++++++++",Time)
    # 阿伦尼乌斯拟合
    # 返回的列表
    #     Time = np.array(abstable['mileage'])
    ret_val = []
    SOH_hat = None
    abstable.to_csv("./Test_1.csv", encoding="utf-8-sig", header=False, index=False)
    if SOH.shape[0] > 10:
        # 有一定的样本点数才有拟合的意义
        # Time = (Time-np.min(Time))/(60*60*24)
        AL = AlFit()
        AL.fit(Time, SOH3)
        # SOH预测
        pre = np.linspace(0, Time[-1], 1000)
        SOH_hat = [AL.arrhenius(x) for x in list(pre)]
        SOH_hat = np.array(SOH_hat)

        # 平均衰减速率
        rate = rate_of_decrease(SOH3, Time)
        ret_val.append(rate)
        (a, b, c) = AL.get_para()
        ret_val.append(a)
        ret_val.append(b)
        ret_val.append(c)
    return pre, SOH_hat




class DataCalculation:
    def __init__(self):
        self.abstable = None

    # 计算平均值用于填充无效数据
    def get_fillval(self,val_list):  
        me = 0
        num = 0
        for val in val_list:
            if np.isnan(val):
                continue
            me += val
            num += 1
        return me / num

    # car = {'车号'：int，'absTime':int,'chargeAi':float,'充电信号':int,'soc':float,'batteryMinTemp':int,'batteryMaxTemp':int}
    def get_singlecar_soh(self, car, InfoFrom=0, df=0, para60=60, capCase=0, constCap=0):
        # InfoFrom 代表df的来源，为1时直接传入
        # car = ['北汽', 'eu5-r500', 1]
        # wpath 为到存放的文件夹为止
        # capCase = 1时使用固定数据的capinit模式

        self.abstable = pd.DataFrame(
            columns=['carNum', 'Time', 'capinit ', 'capinitk', 'cap', 'capk', 'Soh', 'minSOC', 'maxSOC', 'mileage'])

        initsign = 0  # 初始标志
        capsign = 0  # 容量标志位
        capinitk = 0  # 初始容量
        capinit = 0  # 计算初始容量
        capk = 0
        cap = 0  # 计算当前容量
        soh = 0

        # 读取数据并创建数据存放文件夹
        if InfoFrom == 0:
            # data = prepocessing(car)
            data = car
        else:
            data = df

        if data.shape[0] < 10:
            return []
        grouped = data.groupby('充电信号')

        # 开始SOH计算
        chargeMode = -1
        for subgroup in grouped:  # 对每次充电信息解析
            if subgroup[0] > 0:
                subtime = []  # 时间序列
                T_Max = []
                T_Min = []
                T_Max_sum = 0
                T_Min_sum = 0
                # 充电信号
                chargeMode += 1

                subcurrent = []  # 电流序列
                subsoc = []  # SOC序列
                maxsoc = max(subgroup[1]['soc'])

                minsoc = min(subgroup[1]['soc'])
                socgap = maxsoc - minsoc
                Vin = max(subgroup[1]['车号'])

                for temp1 in subgroup[1]['absTime']:
                    subtime.append(temp1)
                mintime = subtime[0]

                if minsoc < 50 and socgap > 20:
                    # need to modify

                    for temp1 in subgroup[1]['batteryMaxTemp']:
                        T_Max.append(temp1)
                    # ModT1 = t1[0]

                    for temp1 in subgroup[1]['batteryMinTemp']:
                        T_Min.append(temp1)
                    # ModT2 = t2[0]

                    for counter4 in range(0, len(subtime)):
                        T_Max_sum = T_Max[counter4] + T_Max_sum
                        T_Min_sum = T_Min[counter4] + T_Min_sum

                    T_max_ave = T_Max_sum / len(subtime)
                    T_min_ave = T_Min_sum / len(subtime)
                    T_average = (T_max_ave + T_min_ave) / 2

                    for temp1 in subgroup[1]['chargeAi']:
                        subcurrent.append(temp1)

                    for temp1 in subgroup[1]['soc']:
                        subsoc.append(temp1)
                    Ah = 0  # 累计Ah数
                    current60 = 0

                    # for filling nan
                    current_fill_val = self.get_fillval(subcurrent)

                    if np.isnan(current_fill_val):
                        print('error222')

                    flag_of_60 = 0
                    delta_Ah = [0]
                    for i in range(0, len(subtime) - 1):
                        # time1=datetime.datetime.strptime(subtime[counter3],"%Y-%m-%d %H:%M:%S")
                        # time2=datetime.datetime.strptime(subtime[counter3+1],"%Y-%m-%d %H:%M:%S")
                        time1 = subtime[i]
                        time2 = subtime[i + 1]
                        # gaps=(time2-time1).total_seconds()
                        gaps = (time2 - time1)
                        if np.isnan(subcurrent[i]):
                            subcurrent[i] = current_fill_val
                        if np.isnan(subcurrent[i + 1]):
                            subcurrent[i + 1] = current_fill_val

                        Ah = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / 3600 + Ah
                        delta_Ah.append(Ah)

                        if np.isnan(subcurrent[i]):
                            print('error111\n')
                        #                         if subsoc[counter3]<60.1 and subsoc[counter3]>59.5:
                        if subsoc[i] >= para60:
                            if flag_of_60 == 0:
                                current60 = Ah
                                flag_of_60 = 1

                    if maxsoc < para60:  # 50->60
                        current60 = Ah
                    if initsign < 1:
                        initsign = 1
                        capinit = 100 * current60 / (min(para60, maxsoc) - minsoc)
                        print(current60)
                        print(capinit)
                        if capinit < 80:
                            # 判断条件可改变
                            capinit = 0
                            initsign = 0
                    else:
                        capinitk = (current60 / (min(para60, maxsoc) - minsoc)) * 100
                    print(capinit)
                    if abs(capinitk - capinit) > 0.05 * capinit:
                        capinit = capinit
                    else:
                        capinit = 0.9 * capinit + 0.1 * capinitk
                    # debug
                    if np.isnan(capinit):
                        print('error', minsoc, maxsoc, Ah, len(subtime))

                    if capCase:
                        capinit = constCap
                        capinitk = constCap
                    if maxsoc >= 95 and minsoc < 40:  # 为了29日大数据平台数据修改
                        if capsign < 1:
                            capsign = 1
                            capk = (Ah / socgap) * 100
                            cap = capk
                            soh1 = (capk / capinit) * 100
                        else:
                            capk = (Ah / (maxsoc - minsoc)) * 100
                            if 1.2 * cap > capk > 0.8 * cap:
                                capk = (Ah / (maxsoc - minsoc)) * 100
                                cap = 0.9 * cap + 0.1 * capk
                            else:
                                cap = cap
                            soh1 = (cap / capinit) * 100
                        if np.isnan(T_average):
                            T_average = 25

                        soh = soh1 * (1 - 0.02 * (T_average - 25) / 10)  # 温度修正,修正到25℃
                        if soh >= 100:
                            soh = soh1
                    #                     if soh>100:
                    if np.isnan(soh):
                        print('error', minsoc, maxsoc, Ah, T_average)

                    if capCase:
                        capinit = constCap
                        capinitk = constCap

                # Vin = int(Vin)
                temptable = pd.DataFrame(
                    [Vin, mintime, capinit, capinitk, cap, capk, soh, minsoc, maxsoc, chargeMode]).T
                temptable.columns = ['carNum', 'Time', 'capinit ', 'capinitk', 'cap', 'capk', 'Soh', 'minSOC', 'maxSOC',
                                     'chargeMode']

                self.abstable = pd.concat([self.abstable, temptable], axis=0, sort=False, ignore_index=True)