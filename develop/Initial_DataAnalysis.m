%% 导入数据  （请先修改一下pathname1的路径再运行！！！  pathname1的路径就是“SOH分析”存放的文件夹路径）
clc;
clear;
pathname1='D:\company\SOH分析';  % 修改一下路径！（只需修改这个就行！）
load([pathname1 '\SOH分析\数据\data.mat']); % 导入原始数据库数据 
%% 数据预处理
CarData={};
for CarNumber=1:size(data,2)     % 第几辆车的数据，共55辆车
    A=[];
    C=[];
    E=[];
    G=[];
    ChargeNum=[];
    N_ChrgSts=[];
    ID_ChrgSts=[];
    ID_F=[];
    
    if size(data{1,CarNumber}.user_data,1)==1
        ChargeNum=1; % 充电次数为1
    else 
        ChargeNum=size(data{1,CarNumber}.user_data,1); % 充电次数为data{1,CarNumber}.user_data的行数
    end
    
    for i=1:ChargeNum                                   % 充电次数循环            
        B=data{1,CarNumber}.user_data(i).absTime ;      % 时间戳
        B(:,2)=i;                                       % 充电信号（第几次充电）
        assignin('base',['A',num2str(i)],B);            % 把表1，表2……对应生成A1、A2、A3……
        eval(['A=[A;A',num2str(i),'];']);               % 组合A=[A1;A2;A3;...]     

        D=data{1,CarNumber}.user_data(i).Volt;          % 单体电压（每列）        
        assignin('base',['C',num2str(i)],D);            
        eval(['C=[C;C',num2str(i),'];']); 

        F=data{1,CarNumber}.user_data(i).relTime;       % 绝对时间
        assignin('base',['E',num2str(i)],F);           
        eval(['E=[E;E',num2str(i),'];']);       

        H=data{1,CarNumber}.user_data(i).Info;          % 其他信息
        assignin('base',['G',num2str(i)],H);            
        eval(['G=[G;G',num2str(i),'];']); 
    end
    C(:,end+1)=mean(C,2);       % 求单体电压平均值
    A(:,end+1)=CarNumber;       % 车号
       
%     M_data=[A,C,E,G];
    
    Z=struct2cell(G);           % 把struct格式转换成cell格式         
    Z=cell2mat(Z);              % 把cell格式转换成mat格式
    G_2=Z';
    
    M_data=[A(:,end),A(:,1),E,G_2(:,1:4),A(:,2),G_2(:,5:15),C];  % 重新组合成Excel表格
                                                                 % 每辆车的列数不一样的原因是：每辆车上单体的数量不同
                                                                 % 如果单体数是100个，则表格中共有120列，如果单体数是96个，则表格中共有116列
                                                                 % 最后一列是平均单体电压
                                                                 
    M_data=sortrows(M_data,2);     % 按时间戳进行排序
    ID2=find(M_data(:,13)<-20);    % 过滤掉一些不靠谱的温度数据
    M_data(ID2,:)=[];
    
    M_data(:,2)=M_data(:,2)/1000;
    
    %..................................................生成mat文件............................................................%
     
    Data.CarNumber=M_data(:,1);                 % 车号 
    Data.t_abs=round((M_data(:,2)-min(M_data(:,2))));    % 时间 从0开始
    Data.ChrgSts=M_data(:,8);                   % 充电信号
    Data.date=M_data(:,2);                 % UNIX绝对时间 
    
    N_ChrgSts=unique(Data.ChrgSts);      % 因为之前按照时间戳进行了整体排序，所以充电信号可能是乱的，要重新从1开始编号
    for i=1:size(N_ChrgSts,1)
        ID_ChrgSts=find(Data.ChrgSts==N_ChrgSts(i));
        ID_F(i,1)=min(ID_ChrgSts);
        ID_F(i,2)=max(ID_ChrgSts);
    end
    ID_F=sortrows(ID_F,1);     % 按第一列进行排序
    for j=1:size(ID_F,1)
        Data.ChrgSts(ID_F(j,1):ID_F(j,2))=j;
    end
     
    Data.Tmax=M_data(:,19);                     % Tmax
    Data.Tmin=M_data(:,13);                     % Tmin
    Data.Tmax_No=M_data(:,5);                   % Tmax_No
    Data.Tmin_No=M_data(:,4);                   % Tmin_No
    Data.CellV=M_data(:,20:end-1);              % 所有单体的电压
    Data.CellVmean=M_data(:,end);               % 所有单体的电压的平均值
        
    [max_a,max_sig]=max(Data.CellV,[],2);
    [min_a,min_sig]=min(Data.CellV,[],2);
    
    Data.Vmax=max_a;           % Vmax
    Data.Vmax_No=max_sig;      % Vmax_No
    Data.Vmin=min_a;           % Vmin
    Data.Vmin_No=min_sig;      % Vmin_No
     
    Data.Vzong=sum(Data.CellV,2);     % 总电压
    Data.Current=M_data(:,6);         % 总电流
    Data.SOC=M_data(:,9);             % 总SOC
    
    CarData{CarNumber,1}=Data;

    pathname2=[pathname1 '\SOH分析\结果\'];

    %..................................................生成Excel文件............................................................%
    
    % 待完成问题：如何把时间戳转换成日期格式，然后添加到excel表中,现在只能手动
    
    assignin('base',['CarData',num2str(CarNumber)],M_data);   % 生成每一辆车的数据CarData1，CarData2，CarData3……
    
    filename=['Car' num2str(CarNumber) '-充电' num2str(ChargeNum) '次' '.xlsx',];
    
    pathname3=[pathname1 '\SOH分析\数据\用于估计SOH的数据\'];
    
    MMM={};
    MMM={'车号','absTime','chargeAi','充电信号','soc','batteryMinTemp','batteryMaxTemp'};
    for ii=1:size(M_data,1)
        MMM{ii+1,1}=M_data(ii,1);
        MMM{ii+1,2}=M_data(ii,2);
        MMM{ii+1,3}=M_data(ii,6);
        MMM{ii+1,4}=Data.ChrgSts(ii);
        MMM{ii+1,5}=M_data(ii,9);
        MMM{ii+1,6}=M_data(ii,13);
        MMM{ii+1,7}=M_data(ii,19);
    end
    
    xlswrite([pathname3,filename],MMM)   % 生成Excel表，用于计算SOH
    
    % 记得要把时间戳（第3列）转换成Excel中的时间格式（第2列），=TEXT((C3+8*3600)/86400+70*365+19,"yyyy-mm-dd hh:mm:ss")
    % 以便后期SOH的计算。
    
    disp(['第',num2str(CarNumber) '车' '(共充电',num2str(ChargeNum) '次)的数据已处理完毕！'])

end
%% 保存数据结果
save([pathname2,'All_CarData.mat'],'CarData')   

disp(['共' num2str(size(data,2)) '辆车的数据已处理完毕！！！'])

    

