clc;
clear;
%% ��ȡ�ļ����µ�*.xlsx�ļ� �������޸�һ��pathname1��·�������У�����  pathname1��·�����ǡ�SOH��������ŵ��ļ���·����
pathname1='D:\company\SOH����';  % �޸�һ��·������ֻ���޸�������У���
cd([pathname1 '\SOH����\����\SOH���ƽ��']);  % �л���ǰĿ¼
file=dir('*.xlsx');
file_num=length(file);
for file_counter=1:file_num                                                          
    [text,raw]=xlsread(file(file_counter).name);
    file_headline=raw;
    file_data{file_counter,1}=text;
end
%% ƴ�Ӽ�������  ���ܳ�A
A=[];
for jj=1:file_num 
    B=file_data{jj,1}; 
    A=[A;B];               
end

[m,n]=find(isnan(A));       %ɾ��NaN 
A(m,:)=[];

DataSOH.CarNumber=A(:,1);      % ����
DataSOH.Time=A(:,2);           % ʱ�� ����1970-1-1 00��00��00��ʼ��ָ��ʱ���ʱ��s����
DataSOH.Capinitk=A(:,4);       % Capinitk
DataSOH.Capinit=A(:,3);        % Capinit
DataSOH.Capk=A(:,6);           % Capk
DataSOH.Cap=A(:,5);            % Cap
DataSOH.Soh=A(:,7);            % Soh
DataSOH.minSOC=A(:,8);         % minSOC
DataSOH.maxSOC=A(:,9);         % maxSOC

pathname='D:\company\SOH����\SOH����\���\';

save([pathname,'SOH_Result.mat'],'DataSOH')