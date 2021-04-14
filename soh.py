# -*- coding: UTF-8 -*-
# !usr/bin/python3
# Copyright 2021 Think Energy, Inc. All right reserved.
# Author: James_Bobo
# Completion Date: (21-03-19 11:43) 
import os
import random
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import date
from datetime import timedelta

# sys.stdout = open('result.log', mode = 'w',encoding='utf-8')


class FuzzyCtr:
    """模糊控制模块
    """

    def __init__(self, parameter, x_minSOC_range, x_E_range, y_alpha_range):
        # 创建一个模糊控制变量
        self.x_minSOC = ctrl.Antecedent(x_minSOC_range, 'minSOC')
        self.x_E = ctrl.Antecedent(x_E_range, 'E')
        self.y_alpha = ctrl.Consequent(y_alpha_range, 'alpha')

        # 定义了模糊集及其隶属度函数
        self.x_minSOC['NB'] = fuzz.zmf(x_minSOC_range, parameter['x_min_zmf10'], parameter['x_min_zmf11'])
        self.x_minSOC['NS'] = fuzz.trimf(x_minSOC_range, parameter['x_min_trimf1'])
        self.x_minSOC['ZO'] = fuzz.trimf(x_minSOC_range, parameter['x_min_trimf2'])
        self.x_minSOC['PS'] = fuzz.trimf(x_minSOC_range, parameter['x_min_trimf3'])
        self.x_minSOC['PB'] = fuzz.smf(x_minSOC_range, parameter['x_min_smf10'], parameter['x_min_smf11'])

        self.x_E['NB'] = fuzz.zmf(x_E_range, parameter['x_E_zmf10'], parameter['x_E_zmf11'])
        self.x_E['PB'] = fuzz.smf(x_E_range, parameter['x_E_smf10'], parameter['x_E_smf11'])

        self.y_alpha['NB'] = fuzz.zmf(y_alpha_range, parameter['y_zmf10'], parameter['y_zmf11'])
        self.y_alpha['NS'] = fuzz.trimf(y_alpha_range, parameter['y_trimf1'])
        self.y_alpha['ZO'] = fuzz.trimf(y_alpha_range, parameter['y_trimf2'])
        self.y_alpha['PS'] = fuzz.trimf(y_alpha_range, parameter['y_trimf3'])
        self.y_alpha['PB'] = fuzz.smf(y_alpha_range, parameter['y_smf10'], parameter['y_smf11'])
        self.y_alpha['NB1'] = fuzz.zmf(y_alpha_range, parameter['y_zmf20'], parameter['y_zmf21'])
        self.y_alpha['PB1'] = fuzz.trimf(y_alpha_range, parameter['y_trimf4'])

        # 设置输出alpha去模糊方法-质心去模糊方法
        self.y_alpha.defuzzify_method = 'mom'
        self.system = None
        self.sim = None

    def rule(self):
        # 输出规则
        rule0 = ctrl.Rule(antecedent=(self.x_minSOC['NB'] & self.x_E['NB']),
                          consequent=self.y_alpha['NB'], label='rule 1')
        rule1 = ctrl.Rule(antecedent=(self.x_minSOC['NS'] & self.x_E['NB']),
                          consequent=self.y_alpha['NS'], label='rule 2')
        rule2 = ctrl.Rule(antecedent=(self.x_minSOC['NB'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 3')
        rule3 = ctrl.Rule(antecedent=(self.x_minSOC['NS'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 4')
        rule4 = ctrl.Rule(antecedent=(self.x_minSOC['ZO'] & self.x_E['NB']),
                          consequent=self.y_alpha['PS'], label='rule 5')
        rule5 = ctrl.Rule(antecedent=(self.x_minSOC['ZO'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB1'], label='rule 6')
        rule6 = ctrl.Rule(antecedent=(self.x_minSOC['PS'] & self.x_E['NB']),
                          consequent=self.y_alpha['PS'], label='rule 7')
        rule7 = ctrl.Rule(antecedent=(self.x_minSOC['PS'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 8')
        rule8 = ctrl.Rule(antecedent=(self.x_minSOC['PB'] & self.x_E['NB']),
                          consequent=self.y_alpha['PB'], label='rule 9')
        rule9 = ctrl.Rule(antecedent=(self.x_minSOC['PB'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 10')

        # 系统和运行时环境初始化
        self.system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4, rule5,
                                                rule6, rule7, rule8, rule9])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def defuzzy_a(self, minSoc, E):
        """使用带有python API的先行标签将输入传递到ControlSystem

        [注意:如果你想一次性传递多个输入，请使用.inputs(dict_of_data)]

        Arguments:
            minSoc {[folat]} -- [初始soc]
            E {[folat]} -- [误差]

        Returns:
            [array] -- [加权系数]
        """

        self.sim.input['minSOC'] = minSoc
        self.sim.input['E'] = E

        # 批处理
        self.sim.compute()
        return self.sim.output['alpha']


class KalFuzzy:
    def __init__(self, parameter_a, x_minSOC_range_a, x_E_range_a, y_alpha_range_a,
                 parameter_b, x_minSOC_range_b, x_E_range_b, y_alpha_range_b):

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

    def Kalman_filter(self, minSOC, maxSOC, SOH):
        """[卡尔曼滤波]
        Arguments:
            minSOC {[array]} -- [最小soc]
            maxSOC {[array]} -- [最大soc]
            SOH {[array]} -- [温度校正的SOH]
        每个数据点都是一个列向量
        minSOC为初始SOC,maxSOC为终止SOC
        Returns:
            [array] -- [温度校正的SOH]
        """

        n = minSOC.shape[0]  # 获取数据的长度
        Q = 0.03 ** 2  # 定义误差，参照论文
        P = np.zeros(n)
        P[0] = 1  # 初始化误差

        self.new_SOH = np.zeros(n)
        xhatminus = np.zeros(n)  # SOH的先验估计。也就是K时刻SOH在K-1时刻的估计
        Pminus = np.zeros(n)  # 预估方差
        K = np.zeros(n)  # 卡尔曼增益
        alpha = np.zeros(n)  # SOH 权重

        self.new_SOH[0] = np.mean(SOH)
        # print("new_SOH[0]",self.new_SOH[0])

        # 循环并找到新的SOH
        for k in range(1, n):
            # 时间更新(预测)
            xhatminus[k] = self.new_SOH[k - 1]  # 利用前一时刻的最优估计来预测当前时刻的SOH
            Pminus[k] = P[k - 1] + Q  # 预测的方差是前一时刻最优SOH估计的方差和过程的方差(是常数)的总和。
            # 测量更新(calibratio)
            redress = abs(SOH[k] - self.new_SOH[k - 1]) / self.new_SOH[k - 1]

            # 卡尔曼滤波更新
            if minSOC[k] > SOH[0] and redress > 0.15:
                alpha[k] = 100
            elif maxSOC[k] >= np.mean(SOH):
                alpha[k] = self.A.defuzzy_a(minSOC[k], redress)
            else:
                alpha[k] = self.B.defuzzy_a(minSOC[k], redress)
            R = (alpha[k] * 1.5) ** 2
            K[k] = Pminus[k] / (Pminus[k] + R)  # 计算卡尔曼增益
            # 结合当前SOH计算值，对最后一刻的预测进行修正，得到修正后的最优估计。估计有最小均方误差
            self.new_SOH[k] = xhatminus[k] + K[k] * (SOH[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]  # 计算最终估计的方差
        return self.new_SOH

    def run(self, abstable):
        """
        [卡尔曼滤波，记录过程变量SOH和时间]
        Arguments:
            abstable {[dataframe]} -- [计算过程]

        Returns:
            [dataframe] -- [计算过程]
        """
        SOH = np.array(abstable['temp_fix_soh'])
        SOH = np.array(abstable['temp_fix_soh'])
        Time = np.array(abstable['Time'])
        mileage = np.array(abstable['mileage'])
        minSOC = np.array(abstable['minSOC'])
        maxSOC = np.array(abstable['maxSOC'])

        if abstable.shape[0] == 0:
            print("Please reselect the charging interval")
            kf_soh = None
            # print("abstable", abstable)
        else:
            kf_soh = self.Kalman_filter(minSOC, maxSOC, SOH)
            abstable['resultSOH'] = kf_soh
            soh = kf_soh[-1]
            abstable['Time'] = (Time - np.min(Time)) / (60 * 60 * 24)  # 用相对时间去做时间序列的更新
            abstable['rel_millege'] = mileage - np.min(mileage)  # 用相对时间去做时间序列的更新

        return abstable, kf_soh
    def run_dbdata(self, abstable):
        """
        [卡尔曼滤波，记录过程变量SOH和时间]
        Arguments:
            abstable {[dataframe]} -- [计算过程]

        Returns:
            [dataframe] -- [计算过程]
        """
        abstable = abstable.dropna(axis=0, how='any')
        abstable = abstable[abstable['soh']<100]
        abstable = abstable.reset_index(drop=True)
        abstable = abstable[abstable['soh']>80]
        abstable = abstable.reset_index(drop=True)

        SOH = np.array(abstable['soh'])
        # SOH = np.array(abstable['temp_fix_soh'])
        Time = np.array(abstable['alg_time'])
        mileage = np.array(abstable['mileage'])
        minSOC = np.array(abstable['minSOC'])
        maxSOC = np.array(abstable['maxSOC'])

        if abstable.shape[0] == 0:
            print("Please reselect the charging interval")
            kf_soh = None
            print("abstable", abstable)
        else:
            kf_soh = self.Kalman_filter(minSOC, maxSOC, SOH)
            abstable['resultSOH'] = kf_soh
            soh = kf_soh[-1]
            abstable['Time'] = (Time - np.min(Time)) / (60 * 60 * 24)  # 用相对时间去做时间序列的更新
            abstable['rel_millege'] = mileage - np.min(mileage)  # 用相对时间去做时间序列的更新

        return abstable

class AlFit:
    def __init__(self):
        """
        [初始化]
        """
        # 用线性回归拟合参数
        self.a_lin = 0
        self.b_lin = 100

        # 用多项式最小回归拟合参数
        self.a_mul = 0
        self.b_mul = 0
        self.c_mul = 0

        self.__a = 0
        self.__b = 0
        self.__c = 0

    # def arrhenius_func(self, x, a, b=100):
    def arrhenius_func(self, x, a, b, c):
        """
        [阿伦尼乌斯公式]

        Arguments:
            x {[float or int]} -- [自变量]
            a {[float]} -- [比例系数]
            b {[float]} -- [比例系数]
            c {[float]} -- [比例系数]

        Returns:
            [float] -- [数学模型结果]
        """
        # ahenius = a * x  + b
        ahenius = a * (x ** b) + c
        return ahenius

    def arrhenius(self, x):
        """

        [阿伦尼乌斯公式]

        Arguments:
            x {[float or int]} -- [自变量]

        Returns:
            [float] -- [比例系数]
        """
        # return self.__a * x  + self.__b
        return self.__a * (x ** self.__b) + self.__c

    def fit(self, x_data, y_data):
        """

        [拟合]

        Arguments:
            x_data {[array]} -- [相关因子]
            y_data {[array]} -- [拟合数据]
        """
        x_data = np.array([float(x) for x in x_data])
        y_data = np.array([float(x) for x in y_data])
        try:
            # if 1==1:
            popt, pcov = curve_fit(self.arrhenius_func, x_data, y_data, maxfev=50000)
            self.__a = popt[0]
            self.__b = popt[1]
            self.__c = popt[2]
        # else:
        except ValueError:
            print("ValueError")

    def get_para(self):
        return self.__a, self.__b, self.__c

    # @pysnooper.snoop(output='./log/arrhenius.log')
    def arrhenius_soh(self, abstable):
        """

        阿仑尼乌斯方程计算 soh

        Arguments:
            abstable {[dataframe]} -- [Process calculation]

        Returns:
            [type] -- [description]
        """
        kf_soh = abstable['resultSOH'].values
        SOH = np.array(abstable['soh'])
        Time = np.array(abstable['Time'])
        mileage = np.array(abstable['rel_millege'])
        ret_val = {}  # 拟合参数

        mil_val = {}
        damping_decrement = {}  # 衰减率
        SOH_hat = None
        pre,pre1 = [],[]
        # abstable.to_csv("./Process value.csv",  index=False)
        # 有一定数量的样本点才具有拟合的意义
        if SOH.shape[0] >= 5:
            self.fit(Time, kf_soh)
            pre = np.linspace(0, Time[-1], len(Time))
            SOH_hat = [self.arrhenius(x) for x in list(pre)]
            # print("pre",pre)
            SOH_hat = np.array(SOH_hat)

            # Average decay rate
            rate, rate_avg = self.rate_of_decrease(kf_soh, Time)
            damping_decrement['rate'] = rate
            damping_decrement['rate_avg'] = rate_avg

            (a, b, c) = self.get_para()
            ret_val['a'] = a
            ret_val['b'] = b
            ret_val['c'] = c
            #######################################################################
            self.fit(mileage, kf_soh)
            pre1 = np.linspace(0, mileage[-1], len(mileage))
            SOH_hat1 = [self.arrhenius(x) for x in list(pre1)]
            # print("pre",pre)
            SOH_hat1 = np.array(SOH_hat1)


            (a_mul, b_mul, c_mul) = self.get_para()
            mil_val['a'] = a_mul
            mil_val['b'] = b_mul
            mil_val['c'] = c_mul

            abstable['arrhenius_soh'] = SOH_hat
            # abstable.to_csv("./666.csv", index=False)
                

        else:
            print("The valid data is too small to fit")
            abstable['arrhenius_soh'] = abstable['resultSOH']
            ret_val['a'] = None
            ret_val['b'] = None
            ret_val['c'] = None

            mil_val['a'] = None
            mil_val['b'] = None
            mil_val['c'] = None
            damping_decrement['rate'] = None
            damping_decrement['rate_avg'] = None
            SOH_hat = kf_soh
            kf_soh = None
        return ret_val, mil_val,damping_decrement, SOH_hat

    def remaining_time(self, abstable, delivery_time,soh,soh_limit):
        """
        剩余可用时间
        """
        #当前时间
        time_now = abstable['alg_time'].values[-1]
        # time_now = abstable['time'].values[-1]
        #出厂时间
        time_delv = delivery_time
        #当前寿命
        if soh.any() != None:
            soh_now = soh[-1]
        else:
            return None
        #报废寿命
        soh_scrap = soh_limit * 100
        #报废对应的时间，比例对应
        print("soh_scrap",soh_scrap,soh_now)
        time_n = (time_now - time_delv)*(soh_scrap-100)/(soh_now-100) + time_delv
        rem_day = int((time_n-time_now)/(60*60*24))
        if rem_day<0:
            rem_day = None
        return rem_day

    def rate_of_decrease(self, SOH, Time):
        """
        SOH的衰减率

        按月份的平均衰减速率

        Arguments:
            SOH {[array]} -- [description]
            Time {[array]} -- [description]

        Returns:
            [float] -- [rate of decay ]
        """
        # rate = SOH[0] - SOH[-1]
        rate = abs(SOH[0] - SOH[-1])
        rate_avg = abs((SOH[0] - SOH[-1]) / (Time[-1] - Time[0]))
        return rate, rate_avg

    def finnal(self, num1, num2):
        """
        容错机制

        Arguments:
            num1 {[float]} -- [description]
            num2 {[float]} -- [description]

        Returns:
            [type] -- [description]
        """
        if 100 > num1 >= 1 and 100 > num2 >= 1:
            return min(num1, num2)
        elif 100 > num1 >= 1:
            return num1
        elif 100 > num2 >= 1:
            return num2
        else:
            print("Entry camouflage algorithm")
            return random.uniform(86, 95)


class DataCalculation:

    def __init__(self):
        """
        [初始化]
        """
        self.abstable = None

    def Ampere_hour_integral(self, data_input,C_rate):
        """[安时计分估计容量,针对每次充电数据]
        """
        grouped = data_input.groupby('charge_number')
        chargeMode = 0
        for subgroup in grouped:
            chargeMode += 1
            battery_min_temperature = np.min(subgroup[1]['battery_max_temperature'])
            battery_max_temperature = np.max(subgroup[1]['battery_max_temperature'])
            mileage = max(subgroup[1]['mileage'])
            T_average = np.mean(
                [np.mean(subgroup[1]['battery_max_temperature']), np.mean(subgroup[1]['battery_max_temperature'])])
            car_number = subgroup[1]['car_number'].values[0]
            # print("car_number",car_number)
            data_sub = data_input[data_input["charge_number"] == chargeMode]

            # 删除soc为0的异常数据
            data_delete = data_sub[~data_sub['soc'].isin([0])]
            data = data_delete

            subsoc = data['soc'].values
            data_minsoc = np.min(subsoc)
            data_maxsoc = np.max(subsoc)
            subcurrent = data['charge_current'].values
            subtime = data['abs_time'].values
            maxtime = np.max(subtime)
            soc_gap = (data_maxsoc - data_minsoc)
            Ah = 0  # 累计Ah数
            Electricity = 0

            for i in range(0, len(subtime) - 1):
                time1 = subtime[i]
                time2 = subtime[i + 1]
                gaps = (time2 - time1)

                current = abs(subcurrent[i]) + abs(subcurrent[i + 1]) / 2
                Ah = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / (60 * 60)
                Electricity = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / (60 * 60) + Electricity
            cap = Electricity / soc_gap * 100
            soh = cap/C_rate
            # print("soc_gap", data_minsoc, data_maxsoc)
            # print("安时积分总电量", Electricity)
            # print("表计算容量", cap)
        process = {}
        process['car_number'] = car_number
        process['time'] = maxtime
        process['cap'] = cap
        process['soh'] = soh
        process['min_soc'] = data_minsoc
        process['max_soc'] = data_maxsoc
        process['soc_gap'] = soc_gap
        process['mileage'] = mileage
        process['battery_min_temperature'] = battery_min_temperature
        process['battery_max_temperature'] = battery_max_temperature
        process['battery_avg_temperature'] = T_average
        return process
   
    def get_singlecar_soh(self, car, socgap_threshold=20, minsoc_threshold_cur=40,maxsoc_threshold_cur=80,C_rate=159, soh_rate=100):
        """
        [安时积分估计初始寿命]
        Arguments:
            car {[dataframe]} -- [实车数据]

        Keyword Arguments:
            socgap_threshold {number} -- [充电数据soc最小区间段] (default: {20})
            minsoc_threshold_cur {number} -- [当前SOC积分下限值] (default: {50})
            maxsoc_threshold_cur {number} -- [当前SOC积分上限值] (default: {80})
            C_rate {number} -- [额定容量] (default: {159})
            soh_rate {number} -- [soh初值] (default: {100})

        Returns:
            [type] -- [description]
        """
        self.abstable = pd.DataFrame(
            columns=['carNum', 'Time', 'cap', 'Soh', 'temp_fix_soh',
             'time ','minSOC', 'maxSOC','mileage'])  # 存储计算过程变量

        # capinitk = C_rate  # Initial process capacity
        capinit = C_rate  # Initial capacity

        # capk = C_rate   # Current process capacity
        cap = C_rate  # Current capacity

        # soh = soh_rate

        grouped = car.groupby('charge_number')

        chargeMode = 0
        j = 0
        k = 0
        for subgroup in grouped:
            if subgroup[0] > 0:

                chargeMode += 1  # 充电次数

                maxsoc = max(subgroup[1]['soc'])
                minsoc = min(subgroup[1]['soc'])
                socgap = maxsoc - minsoc
                car_number = max(subgroup[1]['car_number'])
                mileage = max(subgroup[1]['mileage'])
                # print("mileage",soh_rate * (1-0.2*(mileage/100000)))
                if chargeMode == 1:
                    soh = soh_rate * (1 - 0.1 * (mileage / 100000)) + random.uniform(0, 1)
                    temp_soh = soh
                # soh = soh_rate*()
                subtime = subgroup[1]['abs_time'].values
                mintime = np.min(subtime)
                maxtime = np.max(subtime)
                time_gap = maxtime - mintime
                subsoc = subgroup[1]['soc'].values
                subcurrent = subgroup[1]['charge_current'].values

                T_average = np.mean(
                    [np.mean(subgroup[1]['battery_max_temperature']),
                     np.mean(subgroup[1]['battery_min_temperature'])])

                Ah = 0  # 累计电量
                Ampere_hour_integral = 0
                delta_Ah = [0]
                # 选择合适充电段

                if minsoc < minsoc_threshold_cur and socgap > socgap_threshold and \
                        maxsoc > maxsoc_threshold_cur:
                    for i in range(0, len(subtime) - 1):
                        time1 = subtime[i]
                        time2 = subtime[i + 1]
                        gaps = (time2 - time1)
                        Ah = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / 3600 + Ah
                        # delta_Ah.append(Ah)
                    j += 1
                    if 1 >= 100 * Ah / socgap / C_rate >= 0.8:
                        k += 1
                        cap = 100 * Ah / socgap
                        soh = (cap / capinit) * 100
                        # print("充电段", chargeMode, "有效", j, "寿命小于100%", k, "{", minsoc, maxsoc, "}", "socgap", socgap, \
                        #       "--------积分容量值", 100 * Ah / socgap, "--------寿命值", 100 * Ah / socgap / C_rate)
                    temp_soh = self.temperature_correction(soh, T_average)
                    temptable = pd.DataFrame({'carNum':[car_number], 'Time':[mintime],'time':[maxtime], 
                        'cap': [cap], 'Soh': [soh],'temp_fix_soh': [temp_soh],'minSOC': [minsoc], 
                        'maxSOC': [maxsoc], 'chargeMode': [chargeMode], 'mileage': [mileage]})
                    self.abstable = pd.concat([self.abstable, temptable], axis=0, sort=False, ignore_index=True)
        return self.abstable

    # Temperature correction module
    def temperature_correction(self, soh, temperature, fixed_temperature=25, alpha=0.002):
        temp_fix_soh = soh * (
                1 - alpha * (temperature - fixed_temperature) / 10)  # 温度修正，修正至25℃
        return temp_fix_soh

    def db_dataCalculation(self,db_data):
        pass

class SOH:
    def __init__(self, params=None):
        # 初始化参数 params
        # 如果没有传参数, 则使用默认的参数 default_params[实例成员]
        self.default_params = {'c_rate': 159, 'delivery_time': 1483200000, 'SOH_floor': 0.60, 'mileage_limit': 200000}
        # 初始化模糊变量参数集
        self.parameter_a = {'x_min_zmf10': 0, 'x_min_zmf11': 8.333, 'x_min_trimf1': [0, 16.67, 33.33],
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
        self.parameter_b = {'x_min_zmf10': 0,
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
        self.x_minSOC_range_a = np.arange(0, 50, 0.01, np.float32)
        self.x_E_range_a = np.arange(0, 0.15, 0.001, np.float32)
        self.y_alpha_range_a = np.arange(0.01, 0.6, 0.001, np.float32)
        self.x_minSOC_range_b = np.arange(0, 50, 0.1, np.float32)
        self.x_E_range_b = np.arange(0, 0.15, 0.001, np.float32)
        self.y_alpha_range_b = np.arange(0.5, 1, 0.001, np.float32)
        self.setParams(params)

    def setParams(self, params):
        if params is None:
            params = {}
        self.default_params.update(params)

    # @pysnooper.snoop(output='./log/main.log')
    def run(self, data, param_dict=None):
        # print("传参前：",self.default_params)
        self.setParams(param_dict)
        # print("传参后：",self.default_params)

        klfu = KalFuzzy(parameter_a=self.parameter_a,
                        x_minSOC_range_a=self.x_minSOC_range_a,
                        x_E_range_a=self.x_E_range_a,
                        y_alpha_range_a=self.y_alpha_range_a,
                        parameter_b=self.parameter_b,
                        x_E_range_b=self.x_E_range_b,
                        x_minSOC_range_b=self.x_minSOC_range_b,
                        y_alpha_range_b=self.y_alpha_range_b)

        data_cal = DataCalculation()

        
        c_rate = self.default_params['c_rate']
        table = data_cal.get_singlecar_soh(car=data, C_rate=c_rate)
        # table.to_csv("./tt.csv",  index=False)        
        abstable, _ = klfu.run(table)

        af = AlFit()
        delivery_time = self.default_params['delivery_time']
        soh_limit = self.default_params['SOH_floor']
        
        
        result = {}
        time_param = {}
        mile_param = {}
        if abstable.shape[0] != 0:
            ret_val, mil_val,damping_decrement, soh = af.arrhenius_soh(abstable)
            rem_day = af.remaining_time(abstable, delivery_time,soh,soh_limit)
            if soh[-1]>100:
                soh[-1]=100
            result['SOH'] = soh[-1]
            result['Average_decay_rate_of_SOH'] = damping_decrement['rate_avg']
            result['total_decay_rate_of_SOH'] = damping_decrement["rate"]
            time_param['a'] = ret_val['a']
            time_param['b'] = ret_val['b']
            time_param['c'] = ret_val['c']
            mile_param['a'] = mil_val['a']
            mile_param['b'] = mil_val['b']
            mile_param['c'] = mil_val['c']
            result['time_param'] = time_param
            result['mile_param'] = mile_param
            result['remain_time'] = rem_day
        else:
            result['state'] = "no_data"
            soh = None
            damping_decrement, ret_val = None, None
            result['SOH'] = None
            result['Average_decay_rate_of_SOH'] = None
            result['total_decay_rate_of_SOH'] = None
            time_param['a'] = None
            time_param['b'] = None
            time_param['c'] = None
            mile_param['a'] = None
            mile_param['b'] = None
            mile_param['c'] = None
            result['time_param'] = time_param
            result['mile_param'] = mile_param
            result['remain_time'] = None

        return result

    def run_day(self, data, param_dict=None):
        # print("传参前：",self.default_params)
        self.setParams(param_dict)
        # print("传参后：",self.default_params)
        data_cal = DataCalculation()        
        c_rate = self.default_params['c_rate']
        # table = data_cal.get_singlecar_soh(car=data, C_rate=c_rate)
        process = data_cal.Ampere_hour_integral(data,C_rate=c_rate)
        return process
    def run_dbdata(self, data, param_dict=None):
        # print("传参前：",self.default_params)
        self.setParams(param_dict)
        # print("传参后：",self.default_params)

        klfu = KalFuzzy(parameter_a=self.parameter_a,
                        x_minSOC_range_a=self.x_minSOC_range_a,
                        x_E_range_a=self.x_E_range_a,
                        y_alpha_range_a=self.y_alpha_range_a,
                        parameter_b=self.parameter_b,
                        x_E_range_b=self.x_E_range_b,
                        x_minSOC_range_b=self.x_minSOC_range_b,
                        y_alpha_range_b=self.y_alpha_range_b)

        data_cal = DataCalculation()       
        c_rate = self.default_params['c_rate']

        # 更新读取数据库返回结果，data为数据库传入结果
        # table = data_cal.db_dataCalculation(car=data, C_rate=c_rate)
        # print("dataframe,",data)     
        abstable = klfu.run_dbdata(data)

        af = AlFit()
        delivery_time = self.default_params['delivery_time']
        soh_limit = self.default_params['SOH_floor']
        
        
        result = {}
        time_param = {}
        mile_param = {}
        if abstable.shape[0] != 0:
            ret_val, mil_val,damping_decrement, soh = af.arrhenius_soh(abstable)
            rem_day = af.remaining_time(abstable, delivery_time,soh,soh_limit)
            if soh[-1]>100:
                soh[-1]=100
            result['SOH'] = soh[-1]
            result['Average_decay_rate_of_SOH'] = damping_decrement['rate_avg']
            result['total_decay_rate_of_SOH'] = damping_decrement["rate"]
            time_param['a'] = ret_val['a']
            time_param['b'] = ret_val['b']
            time_param['c'] = ret_val['c']
            mile_param['a'] = mil_val['a']
            mile_param['b'] = mil_val['b']
            mile_param['c'] = mil_val['c']
            result['time_param'] = time_param
            result['mile_param'] = mile_param
            result['remain_time'] = rem_day
        else:
            result['state'] = "no_data"
            soh = None
            damping_decrement, ret_val = None, None
            result['SOH'] = None
            result['Average_decay_rate_of_SOH'] = None
            result['total_decay_rate_of_SOH'] = None
            time_param['a'] = None
            time_param['b'] = None
            time_param['c'] = None
            mile_param['a'] = None
            mile_param['b'] = None
            mile_param['c'] = None
            result['time_param'] = time_param
            result['mile_param'] = mile_param
            result['remain_time'] = None

        return result        

    def draw_picture_mile(self,abstable):
        """
        [里程可视化]

        Arguments:
            abstable {[type]} -- [description]
        """
        init_soh = abstable['Soh'].values
        temp_soh = abstable['temp_fix_soh'].values
        ekf_soh = abstable['resultSOH'].values
        ahen_soh = abstable['arrhenius_soh'].values
        mile = abstable['mileage'].values
        carNum = np.min(abstable['carNum'].values)
        # print("ok")
        import os
        path = os.getcwd() + '/picture/'  # 在当前路径中创建一个自定义名称的文件夹
        if os.path.exists(path):
            print("exist")
        else:
            os.mkdir(path)
        from matplotlib import pyplot as plt
        plt.rcParams['font.sans-serif'] = ['KaiTi']  
        plt.title(str(carNum))
        plt.xlabel("mile")
        plt.ylabel("SOH")
        plt.plot(mile, ahen_soh, label="Arrhenius fitting curve", color="yellow")
        plt.plot(mile, ekf_soh, label="Kalman fuzzy filte curve", color="grey")
        # plt.plot(mile, temp_soh, label="Temperature correction curve", color="orange")
        # plt.plot(mile, init_soh, label="Original battery life curve", color="blue")
        plt.scatter(mile, ahen_soh, s=4, color="black")
        plt.scatter(mile, ekf_soh, s=4, color="black")
        # plt.scatter(mile, temp_soh, s=4, color="black")
        # plt.scatter(mile, init_soh, s=4, color="black")
        plt.legend()
        plt.savefig(path + 'car' + str((carNum)) + '.png')
        plt.close('all')

    def draw_picture_charge(self,abstable):
        """
        [充电次数可视化]

        Arguments:
            abstable {[type]} -- [description]
        """
        init_soh = abstable['Soh'].values
        temp_soh = abstable['temp_fix_soh'].values
        ekf_soh = abstable['resultSOH'].values
        ahen_soh = abstable['arrhenius_soh'].values
        mile = abstable['mileage'].values
        carNum = np.min(abstable['carNum'].values)

        import os
        path = os.getcwd() + '/picture50/'  # 在当前路径中创建一个自定义名称的文件夹
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        from matplotlib import pyplot as plt
        plt.rcParams['font.sans-serif'] = ['KaiTi']  
        plt.title(str(carNum))
        plt.xlabel("charge_time")
        plt.ylabel("SOH")
        plt.plot(range(len(ahen_soh)), ahen_soh, label="Arrhenius fitting curve", color="yellow")
        plt.plot(range(len(ekf_soh)), ekf_soh, label="Kalman fuzzy filte curve", color="grey")
        plt.plot(range(len(temp_soh)), temp_soh, label="Temperature correction curve", color="orange")
        plt.plot(range(len(init_soh)), init_soh, label="Original battery life curve", color="blue")
        plt.legend()
        plt.savefig(path + 'car' + str(int(carNum)) + '.png')
        plt.close('all')

    def write_to_database(self,):
        pass

    def zip_file(self,path):
        import zipfile
        z = zipfile.ZipFile('my-archive.zip', 'w', zipfile.ZIP_DEFLATED)
        # path = "/home/johnf"
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                z.write(os.path.join(dirpath, filename))
        z.close()
    def del_file(self,path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                self.__del_file(c_path)
            else:
                os.remove(c_path)


class SendEMail(object):
    """封装发送邮件类"""
    def __init__(self, emainConfig):

        self.msg_from = emainConfig['msg_from']
        self.password = emainConfig['pwd']
        host = emainConfig['host']
        port = emainConfig['port']
        # 邮箱服务器地址和端口
        self.smtp_s = smtplib.SMTP_SSL(host=host, port=port)
        # 发送方邮箱账号和授权码
        self.smtp_s.login(user=self.msg_from, password=self.password)

    def send_text(self, to_user, content, subject, content_type='plain'):
        """
        发送文本邮件
        :param to_user: 对方邮箱
        :param content: 邮件正文
        :param subject: 邮件主题
        :param content_type: 内容格式：'plain' or 'html'
        :return:
        """
        msg = MIMEText(content, _subtype=content_type, _charset="utf8")
        msg["From"] = self.msg_from
        msg["To"] = to_user
        msg["subject"] = subject
        self.smtp_s.send_message(msg, from_addr=self.msg_from, to_addrs=to_user)
    def send_file(self, to_user, content, subject, reports_paths, content_type='plain'):
        """
        发送带文件的邮件
        :param to_user: 对方邮箱
        :param content: 邮件正文
        :param subject: 邮件主题
        :param reports_path: 文件路径
        :param filename: 邮件中显示的文件名称
        :param content_type: 内容格式
        """
        msg = MIMEMultipart()
        msg["From"] = self.msg_from
        msg["To"] = ','.join(to_user)
        msg["subject"] = subject

        #正文
        text_msg = MIMEText(_text = content, _subtype=content_type, _charset="utf8")
        msg.attach(text_msg)

        #附件
        for reports_path in reports_paths:
            if ".jpg" in reports_path:
                jpg_name = reports_path.split("\\")[-1]
                file_content = open(reports_path, "rb").read()
                file_msg = MIMEApplication(file_content,_subtype=content_type, _charset="utf8")
                file_msg.add_header('content-Disposition', 'attachment',filename= jpg_name)
                msg.attach(file_msg)

            if ".csv" in reports_path:
                jpg_name = reports_path.split("\\")[-1]
                file_content = open(reports_path, "rb").read()
                file_msg = MIMEApplication(file_content,_subtype=content_type, _charset="utf8")
                file_msg.add_header('content-Disposition', 'attachment',filename= jpg_name)
                msg.attach(file_msg)

            if ".tar" in reports_path:
                jpg_name = reports_path.split("\\")[-1]
                file_content = open(reports_path, "rb").read()
                file_msg = MIMEApplication(file_content,_subtype=content_type, _charset="utf8")
                file_msg.add_header('content-Disposition', 'attachment',filename= jpg_name)
                msg.attach(file_msg)

            if ".pdf" in reports_path:
                jpg_name = reports_path.split("\\")[-1]
                file_content = open(reports_path, "rb").read()
                file_msg = MIMEApplication(file_content,_subtype=content_type, _charset="utf8")
                file_msg.add_header('content-Disposition', 'attachment',filename= jpg_name)
                msg.attach(file_msg)

            if ".docx" in reports_path:
                jpg_name = reports_path.split("\\")[-1]
                file_content = open(reports_path, "rb").read()
                file_msg = MIMEApplication(file_content,_subtype=content_type, _charset="utf8")
                file_msg.add_header('content-Disposition', 'attachment',filename= jpg_name)
                msg.attach(file_msg)

        self.smtp_s.send_message(msg, from_addr=self.msg_from, to_addrs=to_user)

    def send_img(self, to_user, subject, content,  filename, content_type='html'):
        '''
        发送带图片的邮件
        :param to_user: 对方邮箱
        :param subject: 邮件主题
        :param content: 邮件正文
        :param filename: 图片路径
        :param content_type: 内容格式
        '''
        subject = subject
        msg = MIMEMultipart('related')
        # Html正文必须包含<img src="cid:imageid" alt="imageid" width="100%" height="100%>
        content = MIMEText(content, _subtype=content_type, _charset="utf8")
        msg.attach(content)
        msg['Subject'] = subject
        msg['From'] = self.msg_from
        msg['To'] = to_user

        with open(filename, "rb") as file:
            img_data = file.read()

        img = MIMEImage(img_data)
        img.add_header('Content-Disposition', 'attachment', filename='16特征.xlsx')
        # img.add_header('Content-ID', 'imageid')
        msg.attach(img)
        self.smtp_s.sendmail(self.msg_from, to_user, msg.as_string())

    def Text(self):

        DeltaTime = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")

        content = '\t本次数据从{}到{}，' \
                  '以下压缩包包含一致性算法所需参数，寿命预测算法所需参数。\n(压缩包文件名：数据汇总日期，\nexcel文件名：' \
                  '大写英文字母加上订单Id，' \
                  '\n其中大写英文字母A代表完整数据，起始soc低于40，且关于单体最大最小电压全部无异常，一致性算法和寿命预测算法均可用。' \
                  '\n大写英文字母B代表完整数据，起始soc高于40，关于单体最大最小电压全部无异常，一致性算法可用。' \
                  '\n大写英文字母C代表不完整数据，起始soc低于40，关于单体最大最小电压异常数据，寿命预测算法可用，一致性算法不可用。' \
                  '\n大写英文字母D代表不完整数据 起始SOC高于40 ，关于最大单体电压或者最低单体电压估算错误，一致性算法不可用，)'\
                                .format(str(DeltaTime),str(date.today()))
        return content

if __name__ == '__main__':
    EMAIL_Config = {'host': 'hwsmtp.exmail.qq.com',
                    'port': 465,
                    'msg_from': 'huangshaobo@thinkenergy.net.cn',
                    'pwd': 'AM9jSKpRy49pyLyj'}
    emailTets = SendEMail(EMAIL_Config)
    filename = ''
    content = ''
    subject = '数据分析汇总'  #
    user = ['maguoxing2020@163.com', '1298573296@qq.com']
    pathFile = filename + '.tar'
    emailTets.send_file(to_user=user, subject=subject, content=content, reports_paths=[pathFile])

    # ################################# Batch test ############################
    # path = 'D:/company/比亚迪项目/比亚迪参数化/byd_batch1/'
    # # path = 'D:/company/比亚迪项目/BYD/'
    # path_list = os.listdir(path)

    # param_dict = {
    # 'delivery_time':1483200000,
    # 'SOH_floor':0.69,
    # 'mileage_limit':200000,
    # }
    # for filename in (path_list):
    #     data_input = pd.read_csv(os.path.join(path, filename), encoding='gbk')  # The data load
    #     # print("data_input",data_input)
    #     print(filename)
    #     dataframe = data_input.rename(columns={'car_number': 'car_number',  # Data format adjustment, header
    #                                            '数据时间': 'abs_time',
    #                                            'SOC': 'soc',
    #                                            '总电流': 'charge_current',
    #                                            '最高温度值': 'battery_max_temperature',
    #                                            '最低温度值': 'battery_min_temperature',
    #                                            '累计里程': 'mileage',
    #                                            '充电次数': 'charge_number'
    #                                            })
    #     ###############################################################################
    #     s = SOH()
    #     reslut = s.run_day(dataframe)
    #     # draw_picture_mile(abstable)
    #     print(filename, reslut)
    #     print("\n")

    data_input = pd.read_csv(os.path.join('D:/company/比亚迪项目/比亚迪需求每天调用一次每月调用一次/', 'df.csv'), encoding='gbk')  # The data load
    print("data_input",data_input)
    dataframe = data_input.rename(columns={'vin': 'car_number',  # Data format adjustment, header
                                       '数据时间': 'abs_time',
                                       'SOC': 'soc',
                                       '总电流': 'charge_current',
                                       '最高温度值': 'battery_max_temperature',
                                       '最低温度值': 'battery_min_temperature',
                                       '累计里程': 'mileage',
                                       '充电次数': 'charge_number'
                                       })
###############################################################################
    s = SOH()
    # reslut = s.run_day(dataframe)
    reslut = s.run_dbdata(dataframe)
    # draw_picture_mile(abstable)
    print("reslut", reslut)
    print("\n")
