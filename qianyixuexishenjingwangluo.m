clear;
clc;
close;

% 导入所需的库
import matlab.io.*
%% 读取文件中数据获取线性拟合关系

% 修改文件路径为 C盘我的文件下的Excel文件路径
filePath = 'C:\Users\Lenovo\Desktop\文章5迁移学习\三天数据.xlsx';

% 读取Excel文件
data = xlsread(filePath ,'sheet5');


% 自变量和因变量
% x = data(1:24, 11); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 1); % #1,   假设 A 是第一列，从行 1 到 25
 
% x = data(1:24, 12); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 2); % 假设 A 是第一列，从行 1 到 25
 
% x = data(1:24, 13); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 3); % 假设 A 是第一列，从行 1 到 25
 
% x = data(1:24, 14); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 4); % 假设 A 是第一列，从行 1 到 25
% 
% x = data(1:24, 15); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 5); % 假设 A 是第一列，从行 1 到 25

% x = data(1:24, 16); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 6); % 假设 A 是第一列，从行 1 到 25

% x = data(1:24, 17); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 7); % 假设 A 是第一列，从行 1 到 25

x = data(1:24, 18); % 假设 K 是第二列，从行 2 到 26
y = data(1:24, 8); % 假设 A 是第一列，从行 1 到 25

% x = data(1:24, 19); % 假设 K 是第二列，从行 2 到 26
% y = data(1:24, 9); % 假设 A 是第一列，从行 1 到 25

% 计算拟合系数
p = polyfit(x, y, 1); % 这里 '1' 表示拟合一次线性关系

% 显示拟合系数
disp(p)

%% 预处理
% 初始化变量
deltaT = linspace(0, 30, 10000)'; % deltaT在0到30之间均匀分布，总共有10000个数据点
k = p(1); % 线性关系系数
b= p(2);
GS = k * deltaT+b; % 根据线性关系生成GS数据

% 创建表格以保存数据
T = table(deltaT, GS);

% 写入Excel文件
writetable(T, 'temperature_data.xlsx')

% 读取数据
data2 = readtable('temperature_data.xlsx');

% 分割数据
cv = cvpartition(size(data2,1),'HoldOut',0.3); % 70%的数据用于训练，30%的数据用于测试
idx = cv.test;
% 训练和测试数据集
dataTrain = data2(~idx,:);
dataTest  = data2(idx,:);

% 定义网络结构
% 这里我们使用一个简单的前馈神经网络，有10个隐藏层，这个值可以根据你的问题进行调整。
% 一般来说，更复杂的问题可能需要更多的隐藏层。对于我们这个简单的线性回归问题，10个隐藏层已经足够了。
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% 设置训练参数
% 我们使用随机梯度下降方法训练我们的网络。BatchSize设置为100，这意味着每次更新权重和偏置时会考虑100个样本。
% BatchSize的选择可以根据你的数据和硬件进行调整。较大的BatchSize可能会导致更稳定的训练，但可能会增加计算时间。
net.trainFcn = 'trainscg';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 6;
net.trainParam.min_grad = 1e-6;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% 训练网络
[net,tr] = train(net, dataTrain.deltaT', dataTrain.GS');

%保存网络
save('C:\Users\Lenovo\Desktop\文章5迁移学习\network_checkpoint.mat', 'net');
% 测试网络
predictions = net(dataTest.deltaT');
perf = perform(net, dataTest.GS', predictions);

% 打印性能
disp(perf)


%% 实际数据加入
% 生成数据集2
% deltaT2 = linspace(0, 50, 10000)'; % deltaT在0到50之间均匀分布，总共有10000个数据点
% GS2 = linspace(0, 70, 10000)'; % GS在0到70之间均匀分布，总共有10000个数据点


deltaT2 = repmat(x, 5, 1); % 将原始数组复制5遍
GS2 = repmat(y, 5, 1); % 将原始数组复制5遍

T2 = table(deltaT2, GS2);
writetable(T2, 'dataset2.xlsx');

% 读取数据集2
data3 = readtable('dataset2.xlsx');

% 继续训练网络,tr2是在继续训练网络时返回的训练记录，它包含了训练过程中的各种信息，如训练误差、验证误差等。
[net2, tr2] = train(net, data3.deltaT2', data3.GS2', 'CheckpointFile', 'network_checkpoint.mat');
save(' .mat', 'net2','tr2');

% 测试网络2
predictions2 = net2(data3.deltaT2');
perf2 = perform(net2, data3.GS2', predictions2);

% 打印性能2
disp(perf2);

% 假设已经训练好的网络为 net
load('C:\Users\Lenovo\Desktop\文章5迁移学习\network_checkpoint.mat');  % 加载已经训练好的网络参数


%% 计算数据
% 输入 deltaT 进行推断
deltaT_new =  input('请输入一个数字: ');  % 新的 deltaT 值
GS_predicted = net(deltaT_new);  % 使用网络进行推断，得到预测的 GS 值

disp(GS_predicted);  % 打印预测的 GS 值