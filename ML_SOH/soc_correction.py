import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#对将0-100%SOC-OCV换算到5-97%范围的SOC-OCV
# 额定容量范围内的OCV定义插值函数

class Soc_correction:
    '''SOC校准'''

    def Soc_mapping(self,socmin,socmax,SOC):
        '''
        根据额定SOC使用范围，获得表显0-100%SOC对应的真实
        :param SOC: 0-100%SOC范围内的SOC
        :param socmin: 使用SOC下限
        :param socmax: 使用SOC上限
        :return: 映射到使用范围内的SOC
        '''
        soc_map=[]
        socd=(socmax-socmin)/100
        for i in SOC:
            soc_map.append(1/socd*i-5/socd)
        return soc_map

    def Inter(self,OCV,SOC,ocv_list):
        '''
        插值法校正SOC
        :param SOC: SOC列表
        :param OCV: SOC对应OCV列表
        :param ocv_list: 待校正点的ocv列表
        :return:估算的soc_list
        '''
        soc_list=[]
        for ocv in ocv_list:
            if ocv in OCV:
                soc=SOC[OCV.index(ocv)]
                soc_list.append(soc)
            elif ocv<OCV[0]:
                soc = SOC[0] - (SOC[1] - SOC[0]) / ((OCV[1] - OCV[0])) * (OCV[0] - ocv)
                soc_list.append(soc)
            elif ocv>OCV[-1]:
                soc = SOC[-1] + (SOC[-1] - SOC[-2]) / ((OCV[-1] - OCV[-2])) * (ocv - OCV[-1])
                soc_list.append(soc)
            else:
                for i in range(len(OCV)-1):
                    if ocv>=OCV[i] and ocv<=OCV[i+1]:
                        soc=SOC[i] + (SOC[i+1] - SOC[i]) / ((OCV[i+1] - OCV[i])) * (ocv - OCV[i])
                        soc_list.append(soc)
                    else:
                        continue
        return soc_list
    def soc_correction(self,SOC,OCV,socmin,socmax,ocv_list):
        soc_cor=self.Inter(OCV,SOC,ocv_list)
        soc_list=self.Soc_mapping(socmin,socmax,soc_cor)
        return soc_list



if __name__ == '__main__':
    SOC = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96, 97, 100]
    OCV = [3.281, 3.508, 3.568, 3.597, 3.622, 3.641, 3.682, 3.756, 3.842, 3.94, 4.05, 4.125, 4.15, 4.175]
    socmin=5
    socmax=97
    ocv_list=[3.28, 3.50, 3.56, 3.59, 3.62, 3.64, 3.68, 3.75, 3.84, 3.4, 4.05, 4.125, 4.1, 4.175,4.18]
    soc_cor=Soc_correction()
    print(soc_cor.soc_correction(SOC,OCV,socmin,socmax,ocv_list))