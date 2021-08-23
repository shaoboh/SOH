%% 导入数据

clc;
clear;
load( 'All_CarData.mat');    % 导入所有车处理过后的数据
load('SOH_Result.mat');      % 导入所有车估计的SOH结果
N_car=size(CarData,1); % 车数

Vin=6;                    % 选择车号（可选择1-N_car中的任一个数字）
%% SOH分析

% 选取车辆数据
ID=find(DataSOH.CarNumber==Vin); % 找到某辆车

Time=(DataSOH.Time(ID)-min(DataSOH.Time(ID)))/3600/24; % 时间：单位为天
%disp(["Time",num2str(Time)])
SOH=DataSOH.Soh(ID);            % SOH初步估计结果
minSOC=DataSOH.minSOC(ID);      % 最小SOC
maxSOC=DataSOH.maxSOC(ID);      % 最大SOC

% 删除一些无效的估计点
% ID_del=find((SOH<=60)|(SOH>=100)); 
% 
% Time(ID_del)=NaN; 
% SOH(ID_del)=NaN;
% minSOC(ID_del)=NaN;
% maxSOC(ID_del)=NaN;

%% 卡尔曼滤波+模糊逻辑 对初步估计的SOH2结果进行修正 得到SOH3

new_SOH = KF_FL(minSOC,maxSOC,Time,SOH);  % KF_FL是卡尔曼滤波+模糊逻辑的函数
  
%% 阿伦尼乌斯模型拟合    
   
Time(1)=0.1;
syms n
cfun = fittype('a*n^z','independent','n','coefficients',{'a','z'});  % fittype是自定义拟合函数 y(n)=a*n^z   
f_AL = fit(Time,new_SOH,cfun); % 根据自定义拟合函数f来拟合数据x，y

% f_AL=fit(Time,new_SOH,'power1');

AL_x=min(Time):0.05:max(Time)+1;
AL_y=f_AL(AL_x);   
    
%% 画图

figure
hold on
% plot(Time,SOH,'o');
plot(Time,new_SOH,'o','color','r');
plot(AL_x,AL_y,'color','b')
hold off
legend('SOH估计','Arrhenius拟合','fontsize',12,'LineWidth',1.8)
xlabel('时间(天)','fontsize',18,'LineWidth',1.8);    
ylabel('SOH(%)','fontsize',18,'LineWidth',1.8);  
ylim([min(new_SOH)-1 max(new_SOH)+1])
plotc;


    