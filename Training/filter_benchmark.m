%% ECCV 2020 Filter Benchmark on TCSC
% David Reixach - IRI(CSIC-UPC) - 19.02.2020
% Process Datasets

%% Initialization

clear all, close all, clc
projectStartup('mcsc')
pythonStartup   % correct PY/MKL incompatibility

dbstop if error

rng('default')

% cd([project,'/Tensor_CSC'])
cd ../Tensor_CSC/Training/

%% Start parpool
poolobj = gcp('nocreate');
if(isempty(poolobj))
    parpool(8);
end

%% Select data
% datasetsPath = '/home/david/Modular/Datasets/CVPR20/';
datasetsPath = '/home/dreixach/Modular/Datasets/CVPR20/';

exp = 2;

nameCell = {'01_city_TCSC_';
            '01_fruit_TCSC_'};

dataCell = {[datasetsPath,'city.mat'];
        [datasetsPath,'fruit.mat'];
        [datasetsPath,'city_fruit_testing.mat']};

data = dataCell{exp};
name = nameCell{exp};
dataTest = dataCell{3};

load2(data,'S','b')
load2(dataTest,'S','btest')

%% run

% b = squeeze(b);
% btest = squeeze(btest);

resTraining = [];
resTesting = [];

PARA.maxiter = 10;
PARA.maxiter_x = 1;
PARA.maxiter_d = 1;
PARA.lambda = 1;

PARA.filter_szx = 11;
PARA.filter_szy = 11;

PARAtest = PARA;
PARA.solveDict = true;
PARAtest.solveDict = false;

PARAtest.maxiter = 1;
PARAtest.maxiter_x = 10;

K_exp = [5,15,25,50,100,200];

for K = 5%K_exp
    [D,Dhat,X,Xhat,Y,Yhat,params_sizes] = load_data2(b,K);
    [~,~,X_test,Xhat_test,Y_test,Yhat_test,params_sizes_test] = load_data2(btest,K);


    PARA.n1 = params_sizes(1);
    PARA.n2 = params_sizes(2);
    PARA.n3 = params_sizes(3);
    PARA.n4 = params_sizes(4);
    PARA.N = params_sizes(5);

    PARAtest.n1 = params_sizes_test(1);
    PARAtest.n2 = params_sizes_test(2);
    PARAtest.n3 = params_sizes_test(3);
    PARAtest.n4 = params_sizes_test(4);
    PARAtest.N = params_sizes_test(5);

% run experiment
% 
% resTraining = [];
% resTesting = [];

% for K = K_exp
    
    PARA.K = K;
    PARAtest.K = K;
    
    t1 = tic;
%     [~,Dhat,R_D] = tensor_trainer(Dhat,Xhat,Yhat,PARA);
    td = toc(t1);    
%     fprintf('\nDone training K: %i! --> Time: %2.2f s\n\n', K, td)
    
    t2 = tic;
    [~,~,R_Z] = tensor_trainer(Dhat,Xhat_test,Yhat_test,PARAtest); 
    tc = toc(t2);    
    fprintf('\nDone testing K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR,R_Z.CR)
    
    R_D.K = K;
    R_Z.K = K;
    
    resTraining = [resTraining, R_D];
    resTesting =  [resTesting, R_Z];

end


%% save

dataPath = [project,'/data/'];

save2([dataPath,name,'TrainResults.mat'],'resTrain','dataset','-noappend')
save2([dataPath,name,'TestResults.mat'],'resTest','dataset','-noappend')

dbclear all
