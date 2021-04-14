import pandas as pd
import numpy as np
import heapq
import time

import math
import operator
data_input = pd.read_csv(r'D:/git/诊断仪2.1版本/20210312/df2.csv')

grouped = data_input.groupby('charge_number')
chargeMode = 0
# arr = []
# for subgroup in grouped:
# #     print(len(subgroup[1]))
#     chargeMode += 1 
#     arr.append(len(subgroup[1]))
# # print(len(arr))

# ##########最大区间及其对应的索引
# max_index, max_number  = max(enumerate(arr), key=operator.itemgetter(1))
# min_index, min_number  = min(enumerate(arr), key=operator.itemgetter(1))
# print(min_index)
# print(min_number)

# ###########查看较大区间及其索引
# re1 = heapq.nlargest(1, arr) #求最大的三个元素，并排序
# re2 = map(arr.index, heapq.nlargest(10, arr)) #求最大的三个索引    nsmallest与nlargest相反，求最小
# print(re1)
# print(list(re2)) #因为re2由map()生成的不是list，直接print不出来，添加list()就行了

# # data_sub = data_input[data_input["充电次数"]==min_index]
# data_sub = data_input[data_input["充电次数"]==199]
# #去重
data = data_input.drop_duplicates(['soc'])

#增加标准时间便于观察时间梯度
# data['标准时间'] = np.array([time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)) for x in data['时间'].values])

subsoc = data['soc'].values
subcurrent = data['charge_current'].values
subtime = data['abs_time'].values

soc_gap = 0.1
Ah = 0  # 累计Ah数
soc = 0
total_ah=0

I =[0]
Ampere_hour_integral = 0
delta_Ah = [1.5]
delta_SOC=[0]
delta_Ah_total=[0]
for i in range(0, len(subtime) - 1):
    time1 = subtime[i]
    time2 = subtime[i + 1]
    gaps = (time2 - time1)

    soc=(abs(subsoc[i]) + abs(subsoc[i + 1])) / 2
    current=(abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2
    Ah = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / 3600
    total_ah = (abs(subcurrent[i]) + abs(subcurrent[i + 1])) / 2 * gaps / 3600+total_ah
    subcurrent
    cap = Ah/soc_gap
    delta_SOC.append(soc)
    delta_Ah.append(cap)
    delta_Ah_total.append(total_ah)
    I.append(current)

delta_Ah = [x*100 for x in delta_Ah]
print("total_ah",total_ah)
#作图
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.figure()   #自定义画布大小(width,height)
plt.figure( figsize=(100,80))
plt.title("广汽数据安时积分效果图")
plt.xlabel("soc")
plt.ylabel("△Q")
plt.axhline(227, color='g', linestyle='--',label="额定容量")
plt.plot(delta_SOC,I ,label="电流" ,color="orange")
plt.scatter(delta_SOC,delta_Ah ,label="安时积分" ,color="r")
plt.plot(delta_SOC,delta_Ah ,label="积分容量" ,color="magenta")
plt.plot(delta_SOC,delta_Ah_total ,label="积分容量累加" ,color="r")
plt.legend()
plt.show()
