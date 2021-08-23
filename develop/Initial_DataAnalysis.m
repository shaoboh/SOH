%% ��������  �������޸�һ��pathname1��·�������У�����  pathname1��·�����ǡ�SOH��������ŵ��ļ���·����
clc;
clear;
pathname1='D:\company\SOH����';  % �޸�һ��·������ֻ���޸�������У���
load([pathname1 '\SOH����\����\data.mat']); % ����ԭʼ���ݿ����� 
%% ����Ԥ����
CarData={};
for CarNumber=1:size(data,2)     % �ڼ����������ݣ���55����
    A=[];
    C=[];
    E=[];
    G=[];
    ChargeNum=[];
    N_ChrgSts=[];
    ID_ChrgSts=[];
    ID_F=[];
    
    if size(data{1,CarNumber}.user_data,1)==1
        ChargeNum=1; % ������Ϊ1
    else 
        ChargeNum=size(data{1,CarNumber}.user_data,1); % ������Ϊdata{1,CarNumber}.user_data������
    end
    
    for i=1:ChargeNum                                   % ������ѭ��            
        B=data{1,CarNumber}.user_data(i).absTime ;      % ʱ���
        B(:,2)=i;                                       % ����źţ��ڼ��γ�磩
        assignin('base',['A',num2str(i)],B);            % �ѱ�1����2������Ӧ����A1��A2��A3����
        eval(['A=[A;A',num2str(i),'];']);               % ���A=[A1;A2;A3;...]     

        D=data{1,CarNumber}.user_data(i).Volt;          % �����ѹ��ÿ�У�        
        assignin('base',['C',num2str(i)],D);            
        eval(['C=[C;C',num2str(i),'];']); 

        F=data{1,CarNumber}.user_data(i).relTime;       % ����ʱ��
        assignin('base',['E',num2str(i)],F);           
        eval(['E=[E;E',num2str(i),'];']);       

        H=data{1,CarNumber}.user_data(i).Info;          % ������Ϣ
        assignin('base',['G',num2str(i)],H);            
        eval(['G=[G;G',num2str(i),'];']); 
    end
    C(:,end+1)=mean(C,2);       % �����ѹƽ��ֵ
    A(:,end+1)=CarNumber;       % ����
       
%     M_data=[A,C,E,G];
    
    Z=struct2cell(G);           % ��struct��ʽת����cell��ʽ         
    Z=cell2mat(Z);              % ��cell��ʽת����mat��ʽ
    G_2=Z';
    
    M_data=[A(:,end),A(:,1),E,G_2(:,1:4),A(:,2),G_2(:,5:15),C];  % ������ϳ�Excel���
                                                                 % ÿ������������һ����ԭ���ǣ�ÿ�����ϵ����������ͬ
                                                                 % �����������100���������й���120�У������������96���������й���116��
                                                                 % ���һ����ƽ�������ѹ
                                                                 
    M_data=sortrows(M_data,2);     % ��ʱ�����������
    ID2=find(M_data(:,13)<-20);    % ���˵�һЩ�����׵��¶�����
    M_data(ID2,:)=[];
    
    M_data(:,2)=M_data(:,2)/1000;
    
    %..................................................����mat�ļ�............................................................%
     
    Data.CarNumber=M_data(:,1);                 % ���� 
    Data.t_abs=round((M_data(:,2)-min(M_data(:,2))));    % ʱ�� ��0��ʼ
    Data.ChrgSts=M_data(:,8);                   % ����ź�
    Data.date=M_data(:,2);                 % UNIX����ʱ�� 
    
    N_ChrgSts=unique(Data.ChrgSts);      % ��Ϊ֮ǰ����ʱ��������������������Գ���źſ������ҵģ�Ҫ���´�1��ʼ���
    for i=1:size(N_ChrgSts,1)
        ID_ChrgSts=find(Data.ChrgSts==N_ChrgSts(i));
        ID_F(i,1)=min(ID_ChrgSts);
        ID_F(i,2)=max(ID_ChrgSts);
    end
    ID_F=sortrows(ID_F,1);     % ����һ�н�������
    for j=1:size(ID_F,1)
        Data.ChrgSts(ID_F(j,1):ID_F(j,2))=j;
    end
     
    Data.Tmax=M_data(:,19);                     % Tmax
    Data.Tmin=M_data(:,13);                     % Tmin
    Data.Tmax_No=M_data(:,5);                   % Tmax_No
    Data.Tmin_No=M_data(:,4);                   % Tmin_No
    Data.CellV=M_data(:,20:end-1);              % ���е���ĵ�ѹ
    Data.CellVmean=M_data(:,end);               % ���е���ĵ�ѹ��ƽ��ֵ
        
    [max_a,max_sig]=max(Data.CellV,[],2);
    [min_a,min_sig]=min(Data.CellV,[],2);
    
    Data.Vmax=max_a;           % Vmax
    Data.Vmax_No=max_sig;      % Vmax_No
    Data.Vmin=min_a;           % Vmin
    Data.Vmin_No=min_sig;      % Vmin_No
     
    Data.Vzong=sum(Data.CellV,2);     % �ܵ�ѹ
    Data.Current=M_data(:,6);         % �ܵ���
    Data.SOC=M_data(:,9);             % ��SOC
    
    CarData{CarNumber,1}=Data;

    pathname2=[pathname1 '\SOH����\���\'];

    %..................................................����Excel�ļ�............................................................%
    
    % ��������⣺��ΰ�ʱ���ת�������ڸ�ʽ��Ȼ����ӵ�excel����,����ֻ���ֶ�
    
    assignin('base',['CarData',num2str(CarNumber)],M_data);   % ����ÿһ����������CarData1��CarData2��CarData3����
    
    filename=['Car' num2str(CarNumber) '-���' num2str(ChargeNum) '��' '.xlsx',];
    
    pathname3=[pathname1 '\SOH����\����\���ڹ���SOH������\'];
    
    MMM={};
    MMM={'����','absTime','chargeAi','����ź�','soc','batteryMinTemp','batteryMaxTemp'};
    for ii=1:size(M_data,1)
        MMM{ii+1,1}=M_data(ii,1);
        MMM{ii+1,2}=M_data(ii,2);
        MMM{ii+1,3}=M_data(ii,6);
        MMM{ii+1,4}=Data.ChrgSts(ii);
        MMM{ii+1,5}=M_data(ii,9);
        MMM{ii+1,6}=M_data(ii,13);
        MMM{ii+1,7}=M_data(ii,19);
    end
    
    xlswrite([pathname3,filename],MMM)   % ����Excel�����ڼ���SOH
    
    % �ǵ�Ҫ��ʱ�������3�У�ת����Excel�е�ʱ���ʽ����2�У���=TEXT((C3+8*3600)/86400+70*365+19,"yyyy-mm-dd hh:mm:ss")
    % �Ա����SOH�ļ��㡣
    
    disp(['��',num2str(CarNumber) '��' '(�����',num2str(ChargeNum) '��)�������Ѵ�����ϣ�'])

end
%% �������ݽ��
save([pathname2,'All_CarData.mat'],'CarData')   

disp(['��' num2str(size(data,2)) '�����������Ѵ�����ϣ�����'])

    

