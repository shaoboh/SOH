import numpy as np
import pandas as pd
import time
import sys
import os

class Charging:
    def Rename(self,file_path,initial_format, target_format):
        '''
        文件重命名，更改后缀格式
        :param file_path: 文件路径
        :param initial_format: 初始文件格式，如xslx
        :param target_format: 重命名文件格式，如csv
        :param n: 原文件后缀长度，如txt为3，xslx为4
        :return: None
        '''
        file_list = []
        for root_dir, sub_dir, files in os.walk(file_path):
            for file in files:
                if file.endswith(".{}".format(initial_format)):
                    # 构造绝对路径
                    ##读取sheet页
                    # pd.read_excel(file_path,sheet_name=None).keys()获取excel表格所有的sheet页名称
                    file_name = os.path.join(root_dir, file)
                    file_list.append(file_name)
        for file in file_list:
            file_csv='{}.{}'.format(file[:-len(initial_format)],target_format)
            os.rename(file,file_csv)
        return '完成文件格式修改'

    def Car_merge(self,file_path,new_path,file_format,column_name,vin_name,encoding='gbk',sep='\n'):
        '''
        将同一辆车的运行数据整合到一个文件中，只保存有效数据列
        :param file_path: 原文件路径
        :param new_path: 新文件路径
        :param file_format: 原文件格式
        :param column_name: 需保留的列名列表
        :param vin_name: 车辆编号列名
        :param encoding: 编码模式
        :param sep: 文件分割符
        :return:
        '''
        file_list = []
        for root_dir, sub_dir, files in os.walk(file_path):
            for file in files:
                if file.endswith(".{}".format(file_format)):
                    # 构造绝对路径
                    ##读取sheet页
                    # pd.read_excel(file_path,sheet_name=None).keys()获取excel表格所有的sheet页名称
                    file_name = os.path.join(root_dir, file)
                    file_list.append(file_name)
        for file in file_list:
            df = pd.read_csv(file, sep=sep, encoding=encoding, low_memory=False)
            vin_list=list(set(df[vin_name]))
            for vin in vin_list:
                df1 = df.loc[df.车辆Vin == vin, column_name]
                #如文件存在则忽略列名，如不存在则保留列名
                if os.path.exists(r'{}\{}.{}'.format(new_path,vin,file_format)):
                    df1.to_csv(r'{}\{}.{}'.format(new_path,vin,file_format), mode='a', encoding=encoding, header=False, index=0)
                else:
                    df1.to_csv(r'{}\{}.{}'.format(new_path,vin,file_format), encoding=encoding, index=0)
        return '完成车辆数据合并'

    def Processing(self,file_path,file_format, column_list, charge_code, encoding='gbk',time_format='%Y-%m-%d %H:%M:%S', rest_I=2):
        '''
        提取每次充电的特征数据
        :param file_path: 文件路径
        :param file_format: 文件类型
        :param column_list: 按顺序输入列名列表，对应中文列名为['时间', '最高电压', '最低电压', '总电压', '最高温度', '最低温度', '充电状态', '总里程', '电流', 'SOC']
        :param charge_code: 充电状态标识, 列表
        :param encoding: 解码方式
        :param time_format: 格式化时间
        :param  rest_I: 等效为静置的电流大小
        :return:
        '''
        n=0
        file_name=[]
        name_label=['time','Vmax','Vmin','totalVolt','Tmax','Tmin','chargestate','totalkm','Curr','SOC']#常规列名

        file_list=[]
        for root_dir,sub_dir,files in os.walk(file_path):
            for file in files:
                if file.endswith(".{}".format(file_format)):
                    #构造绝对路径
                    ##读取sheet页
                    #pd.read_excel(file_path,sheet_name=None).keys()获取excel表格所有的sheet页名称
                    file_name = os.path.join(root_dir, file)
                    file_list.append(file_name)

        if __name__ == '__main__':
            print('车辆数:{}'.format(len(file_list)))
        for file in file_list:
            timestart=time.perf_counter()
            df=pd.read_csv(file,encoding=encoding)
            for i in range(len(name_label)):
                df[name_label[i]] = df[column_list[i]]#将列名改为常规名字
            # print('原始时间格式{}'.format(df.loc[0, 'time']))
            df['time'] = df['time'].str[:-4]  # 如文件时间列有其他字符，可通过字符串截取时间
            # print('截取后时间格式{}'.format(df.loc[0, 'time']))
            df = df.loc[:, name_label]
            df['Vd'] = df['Vmax'] - df['Vmin']
            df['time_datetime']=pd.to_datetime(df['time'], errors='coerce')
            df = df.sort_values(by='time_datetime').reset_index(drop=True)  # 数据按时间排序
            df=df.dropna().reset_index(drop=True)
            df['sec'] = df['time'].apply(lambda x: time.mktime(time.strptime(x, time_format)))  # 时间转为s
            df['timeout'] = df['sec'].drop(labels=0, axis=0).reset_index(drop=True)
            df['time_step'] = (df['timeout'] - df['sec']).fillna(method='ffill')
             #创建存储特征的列表
            file_list=[]#文件名
            time_start_list=[]#充电开始时间
            time_end_list = []  # 充电结束时间
            km_list=[]#运行里程
            time0_rest_list=[]#充电前静置时间小于2A
            time0_vmin_list=[]
            time0_vmax_list = []
            time1_rest_list=[]#充电后静置时间小于2A
            time1_vmin_list=[]
            time1_vmax_list = []


            V0min_list=[]#充电前电压
            V0max_list=[]#充电前电压
            Vd0_list=[]#充电前压差
            V1min_list=[]#充电结束电压
            V1max_list=[]#充电结束电压
            Vd1_list=[]#充电结束压差
            T0min_list=[]#充电开始温度
            T0max_list=[]#充电开始温度
            T1min_list=[]#充电结束温度
            T1max_list=[]#充电结束温度
            SOC0_list=[]#充电起始SOC
            SOC1_list=[]#充电结束SOC
            SOC_loss_list=[]#SOC丢失
            I0_list=[]#充电起始电流
            I1_list=[]#充电结束电流
            Ah_list=[]#充入容量
            Wh_list=[]#充入能量

            #提取充电数据，保留算法需求列
            df_cha=df.loc[df.chargestate.isin(charge_code),:].reset_index(drop=True)

            if df_cha.empty:  # 如果没有充电数据，则跳到下一辆车
                continue
            else:
                # 计算计数时间间隔
                df_cha['sec'] = df_cha['time'].apply(lambda x: time.mktime(time.strptime(x, time_format)))  # 时间转为s
                df_cha['timeout'] = df_cha['sec'].drop(labels=0, axis=0).reset_index(drop=True)
                df_cha['time_step'] = (df_cha['timeout'] - df_cha['sec']).fillna(method='ffill')
                df_cha['soc_step'] = (df_cha['SOC'].drop(labels=0, axis=0).reset_index(drop=True) - df_cha['SOC']).fillna(
                    method='ffill')
                time_step_np=np.array(df_cha['time_step'])
                soc_step_np = np.array(df_cha['soc_step'])
                cha_label = []  # 存放每天数据对应的充电次数
                k = 1
                #判断每次充电
                for step in zip(time_step_np, soc_step_np):
                    if step[0] < 3600 and step[1] >= 0:
                        cha_label.append(k)
                    elif step[0] < 60 and step[1] < 0:
                        cha_label.append(k)
                    else:
                        cha_label.append(k)
                        k += 1
                df_cha['cha_num'] = np.array(cha_label)
                file_cha=file.split('\\')[-1]#提取文件名
                df_cha.to_csv('充电数据贴标签{}.csv'.format(file_cha),encoding='gbk')
                cha_num=max(cha_label)
                if __name__ == '__main__':
                    print('{}共{}次充电'.format(file,cha_num))
                for j in range(2,cha_num):
                    if __name__ == '__main__':
                        print('{}第{}-{}次充电'.format(file,cha_num,j))
                    df_cha1 = df_cha[df_cha['cha_num'] == j ].reset_index(drop=True)
                    df_cha1.loc[df_cha1.shape[0] - 1, 'time_step'] = 0

                    #统计充电过程中SOC丢失数据
                    SOC_loss = 0
                    ls = []
                    ls.extend(df_cha1.index[df_cha1['time_step'] > 60])

                    for m in ls:
                        SOC_loss += df_cha1.loc[m, 'soc_step']
                    # print(j + 1, ls, SOC_loss)
                    df_cha1['time_step'][df_cha1['time_step'] > 60] = 0
                    SOC_loss_list.append(SOC_loss)
                    #充电开始前一刻、充电结束后一刻数据提取
                    print('第{}次充电'.format(set(df_cha1['cha_num'])))
                    id_start = df[df.time == df_cha1['time'].iloc[0]].index[0]#充电开始时
                    print(df_cha1['time'].iloc[0])
                    print('充电开始id: {}'.format(id_start))
                    id0_start=id_start-1#充电前一刻
                    id_end= df[df.time == df_cha1['time'].iloc[-1]].index[0]#充电结束时
                    print('充电结束id: {}'.format(id_end))
                    id1_end=id_end+1#充电结束后一刻

                    #充电前静置时间
                    time0_rest = 0
                    # 电流小于2A时认为是静置状态，定位行车结束id
                    id0=id_start
                    while df.loc[id0 - 1, 'Curr'] > -1*rest_I and df.loc[
                        id0 - 1, 'Curr'] < 2 and id0-1>=0:
                        time0_rest += df.loc[id0 - 1, 'time_step']
                        id0=id0-1
                    time0_vmax = df.loc[id0, 'Vmax']
                    time0_vmin=df.loc[id0, 'Vmin']
                    # 充电后静置时间
                    time1_rest = 0
                    # 电流小于2A时认为是静置状态，定位充电静置后id
                    id1 = id_end
                    # print(df.loc[[df.shape[0]-1],:])
                    if id_end+1<df.shape[0]-1:#防止数据最后一个点为充电结束点
                        while df.loc[id1+1, 'Curr'] > -1*rest_I and df.loc[
                            id1+1, 'Curr'] < 2 and id1 + 1<df.shape[0]-1:
                            time1_rest += df.loc[id1+1, 'time_step']
                            id1 = id1 + 1
                        time1_vmax = df.loc[id1, 'Vmax']
                        time1_vmin = df.loc[id1, 'Vmin']

                    #充电充电容量和能量
                    Ah=-(df_cha1['Curr']*df_cha1['time_step']/3600).sum()
                    Wh=-(df_cha1['Curr']*df_cha1['time_step']*df_cha1['totalVolt']/3600/1000).sum()
                    print(Ah,Wh)

                    file_list.append(file_cha)
                    km_list.append(df.loc[id_end, 'totalkm'])
                    time_start_list.append(df.loc[id_start,'time'])#充电开始时间
                    time_end_list.append(df.loc[id_end, 'time'])  # 充电结束时间
                    SOC0_list.append(df.loc[id_start,'SOC'])#充电起始SOC
                    SOC1_list.append(df.loc[id_end,'SOC'])#充电结束SOC
                    time0_rest_list.append(time0_rest)
                    time0_vmin_list.append(time0_vmin)
                    time0_vmax_list.append(time0_vmax)
                    time1_rest_list.append(time1_rest)
                    time1_vmin_list.append(time1_vmin)
                    time1_vmax_list.append(time1_vmax)
                    V0min_list.append(df.loc[id0_start,'Vmin'])#充电前电压
                    V0max_list.append(df.loc[id0_start,'Vmax'])#充电前电压
                    Vd0_list.append(df.loc[id0_start,'Vd'])#充电前压差
                    V1min_list.append(df.loc[id_end,'Vmin'])#充电结束电压
                    V1max_list.append(df.loc[id_end,'Vmax'])#充电结束电压
                    Vd1_list.append(df.loc[id_end,'Vd'])#充电结束压差
                    T0min_list.append(df.loc[id0_start,'Tmin'])#充电开始温度
                    T0max_list.append(df.loc[id0_start,'Tmax'])#充电开始温度
                    T1min_list.append(df.loc[id_end,'Tmin'])#充电结束温度
                    T1max_list.append(df.loc[id_end,'Tmax'])#充电结束温度
                    I0_list.append(df.loc[id0_start, 'Curr'])  # 充电开始电流
                    I1_list.append(df.loc[id_end, 'Curr'])  # 充电结束电流
                    Ah_list.append(Ah)#充入容量
                    Wh_list.append(Wh)#充入能量

                #提取的充电特征存入文件
                df_eng={}
                df_eng=pd.DataFrame(df_eng)
                df_eng['车辆号']=np.array(file_list)
                df_eng['运行里程']=np.array(km_list)
                df_eng['充电开始时间']=np.array(time_start_list)
                df_eng['充电结束时间'] = np.array(time_end_list)
                df_eng['充电起始SOC']=np.array(SOC0_list)
                df_eng['充电结束SOC']=np.array(SOC1_list)
                df_eng['SOC变化']=df_eng['充电结束SOC']-df_eng['充电起始SOC']
                df_eng['SOC丢失']=np.array(SOC_loss_list)
                df_eng['充电开始电流'] = np.array(I0_list)
                df_eng['充电结束电流'] = np.array(I1_list)
                df_eng['充电前静置时间']=np.array(time0_rest_list)
                df_eng['充电前静置Vmin'] = np.array(time0_vmin_list)
                df_eng['充电前静置Vmax'] = np.array(time0_vmax_list)
                df_eng['充电后静置时间'] = np.array(time1_rest_list)
                df_eng['充电后静置Vmin'] = np.array(time1_vmin_list)
                df_eng['充电后静置Vmax'] = np.array(time1_vmax_list)
                df_eng['充电前Vmin']=np.array(V0min_list)
                df_eng['充电前Vmax']=np.array(V0max_list)
                df_eng['充电前Vd']=np.array(Vd0_list)
                df_eng['充电结束Vmin']=np.array(V1min_list)
                df_eng['充电结束Vmax']=np.array(V1max_list)
                df_eng['充电结束Vd']=np.array(Vd1_list)
                df_eng['充电起始Tmin']=np.array(T0min_list)
                df_eng['充电起始Tmax']=np.array(T0max_list)
                df_eng['充电结束Tmin']=np.array(T1min_list)
                df_eng['充电结束Tmax']=np.array(T1max_list)
                df_eng['充电起始I']=np.array(I0_list)
                df_eng['充电结束I']=np.array(I1_list)
                df_eng['充入容量']=np.array(Ah_list)
                df_eng['充入能量']=np.array(Wh_list)
                if n == 0:
                    df_eng.to_csv('充电特征数据.csv', encoding='gbk',index=0)
                    n += 1
                else:
                    df_eng.to_csv('充电特征数据.csv', mode='a', encoding='gbk', header=False,index=0)
                timeend = time.perf_counter()
                print('文件耗时{}min'.format((timeend-timestart)/60))
            print('充电数据提取完成')
        return df_eng
if __name__ == '__main__':
    charging = Charging()
    column_list = ['sample_ts', 'sin_btry_hist_volt', 'sin_btry_lwst_volt', 'vehl_totl_volt', 'hist_temp_sn',
                   'lwst_temp_sn', 'chrg_state', 'mileage', 'vehl_totl_curnt', 'soc']
    file_path=r'E:\蔚来运行数据'
    initial_format='xls'
    target_format='csv'
    # charging.Rename(file_path, initial_format, target_format)
    # print(charging.Rename(file_path, initial_format, target_format))

    # new_path=r'E:\蔚来运行数据\按车辆组合数据'
    file_format='csv'
    # column_name=['vin','sample_ts','sin_btry_hist_volt','sin_btry_lwst_volt','vehl_totl_volt','btry_pak_hist_temp','btry_pak_lwst_temp','chrg_state','mileage','vehl_totl_curnt','soc']
    # charging.Car_merge(file_path, new_path, file_format, column_name, 'vin')
    # print(charging.Car_merge(file_path, new_path, file_format, column_name, 'vin'))
    new_path = r'E:\蔚来运行数据'
    charge_code=[1]
    charging.Processing(new_path, file_format, column_list, charge_code, encoding='gbk', time_format='%Y-%m-%d %H:%M:%S', rest_I=2)
    print(charging.Processing(new_path, file_format, column_list, charge_code, encoding='gbk', time_format='%Y-%m-%d %H:%M:%S', rest_I=2))

