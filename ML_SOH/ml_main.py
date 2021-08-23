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