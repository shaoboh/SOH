%% �������˲�+ģ���߼� ��KF+FL��
function new_SOH = KF_FL(minSOC,maxSOC,Time,SOH)

% ������ʼ��

x1 = minSOC;      % ��ʼSOC             ********
x2 = maxSOC  ;    % ��ֹSOC             ********
n = length(Time); % ��������n��ʱ��    ********

sz = [n, 1];  % n�У�1��
Q = 0.03^2;   % ��ģ������������w�ķ��
P(1) = 1;     % ����Ϊ1����Ӧ�����̶ȡ�

% ����ģ���߼������㷨

% maxSOC����99ģ�������㷨

a = newfis('fuzzy SOC_Capmax');

a = addvar(a,'input','minSOC',[0,50]);
a = addmf(a,'input',1,'NB','zmf',[0,8.333]);
a = addmf(a,'input',1,'NS','trimf',[0,16.67,33.33]);
a = addmf(a,'input',1,'ZO','trimf',[8.333,25,41.67]);
a = addmf(a,'input',1,'PS','trimf',[16.67,33.33,50]);
a = addmf(a,'input',1,'PB','smf',[41.67,50]);

a = addvar(a,'input','e',[0,0.15]);
a = addmf(a,'input',2,'NB1','zmf',[0,0.1125]);
a = addmf(a,'input',2,'PB1','smf',[0.1125,0.15]);

a = addvar(a,'output','alpha',[0.01,0.6]);
a = addmf(a,'output',1,'NB','zmf',[0.01,0.1575]);
a = addmf(a,'output',1,'NS','trimf',[0.01,0.1575,0.305]);
a = addmf(a,'output',1,'ZO','trimf',[0.1575,0.305,0.4525]);
a = addmf(a,'output',1,'PS','trimf',[0.305,0.4525,0.6]);
a = addmf(a,'output',1,'PB','smf',[0.4525,0.6]);
a = addmf(a,'output',1,'NB1','zmf',[0.01,0.1]);
a = addmf(a,'output',1,'PB1','trimf',[0.541,5.91,11.81]);

% ����ģ������             
rulelist=[1 1 1 1 1;       % �����һ���б�ʾ����e
         1 2 5 1 1;        % ��������б�ʾ���alpha
         2 1 2 1 1;        % ������Ϊ�����Ȩ��
         2 2 5 1 1;        % ������ΪANDģ������(1��ӦAND��2��ӦOR)
         3 1 4 1 1;
         3 2 7 1 1;
         4 1 4 1 1;
         4 2 5 1 1;
         5 1 5 1 1;
         5 2 5 1 1];
a = addRule(a,rulelist);

% ���÷�ģ�����㷨
a1 = setfis(a,'DefuzzMethod','mom');
writeFIS(a1,'SOC_Capmax');
a2 = readfis('SOC_Capmax');

% maxSOCС��99ģ�������㷨

b = newfis('fuzzy SOC_Capmin');

b = addvar(b,'input','minSOC',[0,50]);
b = addmf(b,'input',1,'NB','zmf',[0,8.333]);
b = addmf(b,'input',1,'NS','trimf',[0,16.67,33.33]);
b = addmf(b,'input',1,'ZO','trimf',[8.333,25,41.67]);
b = addmf(b,'input',1,'PS','trimf',[16.67,33.33,50]);
b = addmf(b,'input',1,'PB','smf',[41.67,50]);

b = addvar(b,'input','e',[0,0.15]);
b = addmf(b,'input',2,'NB1','zmf',[0,0.1125]);
b = addmf(b,'input',2,'PB1','smf',[0.1125,0.15]);

b = addvar(b,'output','alpha',[0.5,1]);
b = addmf(b,'output',1,'NB','zmf',[0.5,0.625]);
b = addmf(b,'output',1,'NS','trimf',[0.5,0.625,0.75]);
b = addmf(b,'output',1,'ZO','trimf',[0.625,0.75,0.875]);
b = addmf(b,'output',1,'PS','trimf',[0.75,0.875,1]);
b = addmf(b,'output',1,'PB','smf',[0.875,1]);
b = addmf(b,'output',1,'NB1','zmf',[0.5,0.575]);
b = addmf(b,'output',1,'PB1','trimf',[0.95,5.5,10.5]);

% ����ģ������             
rulelist=[1 1 1 1 1;       % �����һ�б�ʾ����e
         1 2 5 1 1;        % ����ڶ��б�ʾ���alpha
         2 1 2 1 1;        % ������Ϊ�����Ȩ��
         2 2 5 1 1;        % ������ΪANDģ������(1��ӦAND��2��ӦOR)
         3 1 4 1 1;
         3 2 7 1 1;
         4 1 4 1 1;
         4 2 5 1 1;
         5 1 5 1 1;
         5 2 5 1 1];
b = addRule(b,rulelist);

% ���÷�ģ�����㷨
b1 = setfis(b,'DefuzzMethod','mom');
writeFIS(b1,'SOC_Capmin');
b2 = readfis('SOC_Capmin');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z = SOH; % z����SOH���    ********

% ��������г�ʼ��
new_SOH = zeros(sz); % �˲����

P = zeros(sz);       % ������Ƶķ���
xhatminus = zeros(sz); % SOH��������ơ�����k-1ʱ�̣���kʱ��SOH�����Ĺ���
Pminus = zeros(sz);    % ������Ƶķ���
K = zeros(sz);         % ���������档
alpha = zeros(sz);     % SOHȨ��

% ��ʼ����

new_SOH(1) = max(SOH); % ��������ȡƽ��ֵ������ԭʼ���Ʋ����룬���ڴ�ȡ���ֵ  ****

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ģ���������˲� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for k = 2:n
        % ʱ����£�Ԥ�⣩
        xhatminus(k) = new_SOH(k-1); % ����һʱ�̵����Ź���ֵ����Ϊ�Ե�ǰʱ�̵�SOH��Ԥ��
        Pminus(k) = P(k-1)+Q; % Ԥ��ķ���Ϊ��һʱ��SOH���Ź���ֵ�ķ�������̷���֮��
        % �������£�У����
        e1 = abs(z(k)-new_SOH(k-1))/new_SOH(k-1);
        if x2(k) >= 99
            if x1(k) > 50 || e1 > 0.15
                alpha(k) = 100;
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % ���㿨��������
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % ��ϵ�ǰʱSOH����ֵ������һʱ�̵�Ԥ�����У�����õ�У��������Ź��ơ��ù��ƾ�����С������
                P(k) = (1-K(k))*Pminus(k); % �������չ���ֵ�ķ���
            else
                alpha(k) = evalfis([x1(k),e1],a2);   
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % ���㿨��������
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % ��ϵ�ǰʱSOH����ֵ������һʱ�̵�Ԥ�����У�����õ�У��������Ź��ơ��ù��ƾ�����С������
                P(k) = (1-K(k))*Pminus(k); % �������չ���ֵ�ķ���
            end
          
        else
            if x1(k) > 50 || e1 > 0.15
                alpha(k) = 100;
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % ���㿨��������
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % ��ϵ�ǰʱSOH����ֵ������һʱ�̵�Ԥ�����У�����õ�У��������Ź��ơ��ù��ƾ�����С������
                P(k) = (1-K(k))*Pminus(k); % �������չ���ֵ�ķ���
            else
                alpha(k) = evalfis([x1(k),e1],b2);   
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % ���㿨��������
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % ��ϵ�ǰʱSOH����ֵ������һʱ�̵�Ԥ�����У�����õ�У��������Ź��ơ��ù��ƾ�����С������
                P(k) = (1-K(k))*Pminus(k); % �������չ���ֵ�ķ���
            end
        end
    end
end