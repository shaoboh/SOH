%% ��������

clc;
clear;
load( 'All_CarData.mat');    % �������г�������������
load('SOH_Result.mat');      % �������г����Ƶ�SOH���
N_car=size(CarData,1); % ����

Vin=6;                    % ѡ�񳵺ţ���ѡ��1-N_car�е���һ�����֣�
%% SOH����

% ѡȡ��������
ID=find(DataSOH.CarNumber==Vin); % �ҵ�ĳ����

Time=(DataSOH.Time(ID)-min(DataSOH.Time(ID)))/3600/24; % ʱ�䣺��λΪ��
%disp(["Time",num2str(Time)])
SOH=DataSOH.Soh(ID);            % SOH�������ƽ��
minSOC=DataSOH.minSOC(ID);      % ��СSOC
maxSOC=DataSOH.maxSOC(ID);      % ���SOC

% ɾ��һЩ��Ч�Ĺ��Ƶ�
% ID_del=find((SOH<=60)|(SOH>=100)); 
% 
% Time(ID_del)=NaN; 
% SOH(ID_del)=NaN;
% minSOC(ID_del)=NaN;
% maxSOC(ID_del)=NaN;

%% �������˲�+ģ���߼� �Գ������Ƶ�SOH2����������� �õ�SOH3

new_SOH = KF_FL(minSOC,maxSOC,Time,SOH);  % KF_FL�ǿ������˲�+ģ���߼��ĺ���
  
%% ��������˹ģ�����    
   
Time(1)=0.1;
syms n
cfun = fittype('a*n^z','independent','n','coefficients',{'a','z'});  % fittype���Զ�����Ϻ��� y(n)=a*n^z   
f_AL = fit(Time,new_SOH,cfun); % �����Զ�����Ϻ���f���������x��y

% f_AL=fit(Time,new_SOH,'power1');

AL_x=min(Time):0.05:max(Time)+1;
AL_y=f_AL(AL_x);   
    
%% ��ͼ

figure
hold on
% plot(Time,SOH,'o');
plot(Time,new_SOH,'o','color','r');
plot(AL_x,AL_y,'color','b')
hold off
legend('SOH����','Arrhenius���','fontsize',12,'LineWidth',1.8)
xlabel('ʱ��(��)','fontsize',18,'LineWidth',1.8);    
ylabel('SOH(%)','fontsize',18,'LineWidth',1.8);  
ylim([min(new_SOH)-1 max(new_SOH)+1])
plotc;


    