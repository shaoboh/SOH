%% 卡尔曼滤波+模糊逻辑 （KF+FL）
function new_SOH = KF_FL(minSOC,maxSOC,Time,SOH)

% 参数初始化

x1 = minSOC;      % 起始SOC             ********
x2 = maxSOC  ;    % 终止SOC             ********
n = length(Time); % 计算连续n个时刻    ********

sz = [n, 1];  % n行，1列
Q = 0.03^2;   % 本模型中输入噪声w的方差。
P(1) = 1;     % 误差方差为1，反应波动程度。

% 加入模糊逻辑控制算法

% maxSOC大于99模糊控制算法

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

% 建立模糊规则             
rulelist=[1 1 1 1 1;       % 矩阵第一二列表示输入e
         1 2 5 1 1;        % 矩阵第三列表示输出alpha
         2 1 2 1 1;        % 第四列为规则的权重
         2 2 5 1 1;        % 第五列为AND模糊运算(1对应AND，2对应OR)
         3 1 4 1 1;
         3 2 7 1 1;
         4 1 4 1 1;
         4 2 5 1 1;
         5 1 5 1 1;
         5 2 5 1 1];
a = addRule(a,rulelist);

% 设置反模糊化算法
a1 = setfis(a,'DefuzzMethod','mom');
writeFIS(a1,'SOC_Capmax');
a2 = readfis('SOC_Capmax');

% maxSOC小于99模糊控制算法

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

% 建立模糊规则             
rulelist=[1 1 1 1 1;       % 矩阵第一列表示输入e
         1 2 5 1 1;        % 矩阵第二列表示输出alpha
         2 1 2 1 1;        % 第三列为规则的权重
         2 2 5 1 1;        % 第四列为AND模糊运算(1对应AND，2对应OR)
         3 1 4 1 1;
         3 2 7 1 1;
         4 1 4 1 1;
         4 2 5 1 1;
         5 1 5 1 1;
         5 2 5 1 1];
b = addRule(b,rulelist);

% 设置反模糊化算法
b1 = setfis(b,'DefuzzMethod','mom');
writeFIS(b1,'SOC_Capmin');
b2 = readfis('SOC_Capmin');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z = SOH; % z修正SOH结果    ********

% 对数组进行初始化
new_SOH = zeros(sz); % 滤波结果

P = zeros(sz);       % 后验估计的方差
xhatminus = zeros(sz); % SOH的先验估计。即在k-1时刻，对k时刻SOH做出的估计
Pminus = zeros(sz);    % 先验估计的方差
K = zeros(sz);         % 卡尔曼增益。
alpha = zeros(sz);     % SOH权重

% 初始估计

new_SOH(1) = max(SOH); % 理论上是取平均值，由于原始估计不理想，故在此取最大值  ****

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 模糊卡尔曼滤波 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for k = 2:n
        % 时间更新（预测）
        xhatminus(k) = new_SOH(k-1); % 用上一时刻的最优估计值来作为对当前时刻的SOH的预测
        Pminus(k) = P(k-1)+Q; % 预测的方差为上一时刻SOH最优估计值的方差与过程方差之和
        % 测量更新（校正）
        e1 = abs(z(k)-new_SOH(k-1))/new_SOH(k-1);
        if x2(k) >= 99
            if x1(k) > 50 || e1 > 0.15
                alpha(k) = 100;
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % 计算卡尔曼增益
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % 结合当前时SOH计算值，对上一时刻的预测进行校正，得到校正后的最优估计。该估计具有最小均方差
                P(k) = (1-K(k))*Pminus(k); % 计算最终估计值的方差
            else
                alpha(k) = evalfis([x1(k),e1],a2);   
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % 计算卡尔曼增益
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % 结合当前时SOH计算值，对上一时刻的预测进行校正，得到校正后的最优估计。该估计具有最小均方差
                P(k) = (1-K(k))*Pminus(k); % 计算最终估计值的方差
            end
          
        else
            if x1(k) > 50 || e1 > 0.15
                alpha(k) = 100;
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % 计算卡尔曼增益
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % 结合当前时SOH计算值，对上一时刻的预测进行校正，得到校正后的最优估计。该估计具有最小均方差
                P(k) = (1-K(k))*Pminus(k); % 计算最终估计值的方差
            else
                alpha(k) = evalfis([x1(k),e1],b2);   
                R = (alpha(k)*1.5)^2;
                K(k) = Pminus(k)/( Pminus(k)+R ); % 计算卡尔曼增益
                new_SOH(k) = xhatminus(k)+K(k)*(z(k)-xhatminus(k)); % 结合当前时SOH计算值，对上一时刻的预测进行校正，得到校正后的最优估计。该估计具有最小均方差
                P(k) = (1-K(k))*Pminus(k); % 计算最终估计值的方差
            end
        end
    end
end