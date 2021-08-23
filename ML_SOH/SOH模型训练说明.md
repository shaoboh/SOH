# 说明文档 v1.0.0

#### 开发者: 陈娟

##### 代码结构：SOH模型训练

### 

1. 功能说明：输入实车运行数据，返回SOH估算模型。
包含3个py文件：charging.py为充电数据提取程序，soc_correction.py为SOC校正程序, model.py为回归模型训练程序


2. 输入输出参数说明：
（1）charging.py充电数据提取程序参数说明：
输入--文件储存路径，输出--充电特征数据dataframe
如数据文件格式不是csv,则需将文件重命名为csv；如文件不是按vin码拆分的，需按vin码拆分合并，使一辆车的数据保存在一个文件中。

（2）soc_correction.py SOC校正程序参数说明：
输入：
SOC列表（SOC-OCV曲线中的SOC）：list
OCV列表（SOC-OCV曲线中的OCV）：list
使用SOC上限：float
使用SOC下限：float
待校正的电压列表：list

输出：
校正后的SOC列表：list

（3）model.py回归模型训练程序参数说明；
1）建模数据过滤函数Data_filter参数说明：
输入：
file_path：文件储存路径，str
Tmin：筛选充电起始最低温度，默认为20，float
SOCmin：筛选充电结束最高SOC，默认20，float
SOCmax：筛选充电结束最低SOC，默认95，float
dSOC_filter：筛选充电SOC变化不低于dSOC_filter，默认30，float
km_new_filter：筛选行驶里程不超过km_new_filter，默认新车阶段为小于10000，float
rest_time_filter：筛选充电前等效静置（默认<2A为等效静置）的时间不少于rest_time_filter，默认180s，float
abnormal_str：异常字符，默认为[np.inf, -np.inf,'#NAME?']，list
encoding：编码，默认'gbk'，str
输出：
过滤后的dataframe。
2）主程序Model_train参数说明：
输入：
df：Data_filter筛选后的数据，dataframe
column_name：用于建模的数据列选择，list
输出：
输出精度并保存模型


4. how to run：
import charging
import soc_correction
import model

charging = Charging()
column_list = ['sample_ts', 'sin_btry_hist_volt', 'sin_btry_lwst_volt', 'vehl_totl_volt', 'hist_temp_sn',
               'lwst_temp_sn', 'chrg_state', 'mileage', 'vehl_totl_curnt', 'soc']
file_path=r'E:\蔚来运行数据'
initial_format='xls'
target_format='csv'
charging.Rename(file_path, initial_format, target_format)

new_path=r'E:\蔚来运行数据\按车辆组合数据'
file_format='csv'
column_name=['vin','sample_ts','sin_btry_hist_volt','sin_btry_lwst_volt','vehl_totl_volt','btry_pak_hist_temp','btry_pak_lwst_temp','chrg_state','mileage','vehl_totl_curnt','soc']
charging.Car_merge(file_path, new_path, file_format, column_name, 'vin')

new_path = r'E:\蔚来运行数据'
charge_code=[1]
df_cha=charging.Processing(new_path, file_format, column_list, charge_code, encoding='gbk', time_format='%Y-%m-%d %H:%M:%S', rest_I=2)

soc_cor=Soc_correction()
SOC = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96, 97, 100]
OCV = [3.281, 3.508, 3.568, 3.597, 3.622, 3.641, 3.682, 3.756, 3.842, 3.94, 4.05, 4.125, 4.15, 4.175]
socmin=5
socmax=97
ocv_list=[3.28, 3.50, 3.56, 3.59, 3.62, 3.64, 3.68, 3.75, 3.84, 3.4, 4.05, 4.125, 4.1, 4.175,4.18]
soc_cor=Soc_correction()
df_cha['充电前校正SOCmin']=np.array(soc_cor.soc_correction(SOC,OCV,socmin,socmax,list(df_cha['充电前Vmin'])))
df_cha['充电前校正SOCmax']=np.array(soc_cor.soc_correction(SOC,OCV,socmin,socmax,list(df_cha['充电前Vmax'])))

model=Model()
file_path=r'F:\运行数据\A-标准化程序\SOH_model\soh_model\test\蔚来充电特征数据.csv'
model_column=['运行里程', '充电前Vmin', '充电前Vmax', '充电前Vd', '充电结束Vmin', '充电结束Vmax', '充电结束Vd',
                                              '充电起始Tmin','充电起始Tmax', '充电结束Tmin', '充电结束Tmax', '充电起始SOC','充电结束SOC','SOC变化','充入容量']
print(model.Model_train(model.Data_filter(file_path),model_column))


   





