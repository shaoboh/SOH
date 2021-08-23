# -*- coding: UTF-8 -*-
#!usr/bin/python3
#Copyright 2020 Think Team, Inc. All right reserved.
#Author: James_Bobo
#Completion Date: 2020-10-29
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np


# 模糊控制模块
class FuzzyCtr:

    def __init__(self, parameter, x_minSOC_range, x_E_range, y_alpha_range):
        # for fuzzy maxSOC>99%

        # 创建模糊控制变量
        self.x_minSOC = ctrl.Antecedent(x_minSOC_range, 'minSOC')
        self.x_E = ctrl.Antecedent(x_E_range, 'E')
        self.y_alpha = ctrl.Consequent(y_alpha_range, 'alpha')

        # 定义模糊集和其隶属度函数
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

        # 设定输出alpha的解模糊方法——质心解模糊方式
        self.y_alpha.defuzzify_method = 'mom'
        self.system = None
        self.sim = None

    def rule(self):
        # 输出为NB的规则
        rule0 = ctrl.Rule(antecedent=(self.x_minSOC['NB'] & self.x_E['NB']),
                          consequent=self.y_alpha['NB'], label='rule 1')
        # 输出为NS的规则
        rule1 = ctrl.Rule(antecedent=(self.x_minSOC['NS'] & self.x_E['NB']),
                          consequent=self.y_alpha['NS'], label='rule 2')
        # 输出为NS的规则
        rule2 = ctrl.Rule(antecedent=(self.x_minSOC['NB'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 3')
        # 输出为NS的规则
        rule3 = ctrl.Rule(antecedent=(self.x_minSOC['NS'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 4')
        # 输出为NS的规则
        rule4 = ctrl.Rule(antecedent=(self.x_minSOC['ZO'] & self.x_E['NB']),
                          consequent=self.y_alpha['PS'], label='rule 5')
        # 输出为NS的规则
        rule5 = ctrl.Rule(antecedent=(self.x_minSOC['ZO'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB1'], label='rule 6')
        # 输出为NS的规则
        rule6 = ctrl.Rule(antecedent=(self.x_minSOC['PS'] & self.x_E['NB']),
                          consequent=self.y_alpha['PS'], label='rule 7')
        # 输出为NS的规则
        rule7 = ctrl.Rule(antecedent=(self.x_minSOC['PS'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 8')
        # 输出为NS的规则
        rule8 = ctrl.Rule(antecedent=(self.x_minSOC['PB'] & self.x_E['NB']),
                          consequent=self.y_alpha['PB'], label='rule 9')
        # 输出为NS的规则
        rule9 = ctrl.Rule(antecedent=(self.x_minSOC['PB'] & self.x_E['PB']),
                          consequent=self.y_alpha['PB'], label='rule 10')

        # 系统和运行环境初始化
        self.system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def defuzzy_a(self, minSoc, E):
        # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
        # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
        self.sim.input['minSOC'] = minSoc
        self.sim.input['E'] = E

        # Crunch the numbers
        self.sim.compute()
        return self.sim.output['alpha']


if __name__ == "__main__":
    # FuzzyCtr的输入参数
    parameter_a = {'x_min_zmf10': 0, 
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
    x_minSOC_range_a = np.arange(0, 50, 0.01, np.float32)
    x_E_range_a = np.arange(0, 0.15, 0.001, np.float32)
    y_alpha_range_a = np.arange(0.01, 0.6, 0.001, np.float32)

    parameter_b = {'x_min_zmf10': 0, 
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
    x_minSOC_range_b = np.arange(0, 50, 0.1, np.float32)
    x_E_range_b = np.arange(0, 0.15, 0.001, np.float32)
    y_alpha_range_b = np.arange(0.5, 1, 0.001, np.float32)

    A = FuzzyCtr(parameter=parameter_a, x_minSOC_range=x_minSOC_range_a, x_E_range=x_E_range_a,
                 y_alpha_range=y_alpha_range_a)
    B = FuzzyCtr(parameter=parameter_b, x_minSOC_range=x_minSOC_range_b, x_E_range=x_E_range_b,
                 y_alpha_range=y_alpha_range_b)

    # visualization
    A.y_alpha.view()
    B.y_alpha.view()
