clc;
clear;
%% 读取文件夹下的*.xlsx文件 （请先修改一下pathname1的路径再运行！！！  pathname1的路径就是“SOH分析”存放的文件夹路径）
pathname1='D:\company\SOH分析';  % 修改一下路径！（只需修改这个就行！）
cd([pathname1 '\SOH分析\数据\SOH估计结果']);  % 切换当前目录
file=dir('*.xlsx');
file_num=length(file);
for file_counter=1:file_num                                                          
    [text,raw]=xlsread(file(file_counter).name);
    file_headline=raw;
    file_data{file_counter,1}=text;
end
%% 拼接几个个表  汇总成A
A=[];
for jj=1:file_num 
    B=file_data{jj,1}; 
    A=[A;B];               
end

[m,n]=find(isnan(A));       %删除NaN 
A(m,:)=[];

DataSOH.CarNumber=A(:,1);      % 车号
DataSOH.Time=A(:,2);           % 时间 （从1970-1-1 00：00：00开始到指点时间的时间差（s））
DataSOH.Capinitk=A(:,4);       % Capinitk
DataSOH.Capinit=A(:,3);        % Capinit
DataSOH.Capk=A(:,6);           % Capk
DataSOH.Cap=A(:,5);            % Cap
DataSOH.Soh=A(:,7);            % Soh
DataSOH.minSOC=A(:,8);         % minSOC
DataSOH.maxSOC=A(:,9);         % maxSOC

pathname='D:\company\SOH分析\SOH分析\结果\';

save([pathname,'SOH_Result.mat'],'DataSOH')