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
% poolobj = gcp('nocreate');
% if(isempty(poolobj))
%     parpool(8);
% end

%% Select data
% datasetsPath = '/home/david/Modular/Datasets/CVPR20/';
datasetsPath = '/home/dreixach/Modular/Datasets/CVPR20/';

exp = 1;

nameCell = {'02_city_TCSC_';
            '02_fruit_TCSC_';
            '02_caltech_city_TCSC_';
            '02_caltech_fruit_TCSC_'};

dataTrainCell = {[datasetsPath,'city.mat'];
            [datasetsPath,'fruit.mat'];
            [datasetsPath,'city.mat'];
            [datasetsPath,'fruit.mat']};
    
dataTestCell = {[datasetsPath,'city_fruit_testing.mat'];
            [datasetsPath,'city_fruit_testing.mat'];
            [datasetsPath,'caltech_testing.mat'];
            [datasetsPath,'caltech_testing.mat']};

name = nameCell{exp};
data = dataTrainCell{exp};
dataTest = dataTestCell{exp};

load2(data,'S','b')
load2(dataTest,'S','btest')

%% run

cellTest = iscell(btest);

resTraining = [];
resTesting = [];

PARA.maxiter = 30;
PARA.maxiter_x = 10;
PARA.maxiter_d = 10;
PARA.lambda = 1;

PARA.filter_szx = 11;
PARA.filter_szy = 11;

PARAtest = PARA;
PARA.solveDict = true;
PARAtest.solveDict = false;

PARAtest.maxiter = 1;
PARAtest.maxiter_x = 60;

K_exp = [5,15,25,50,100,200];

for K = 100;%K_exp
    [D,Dhat,X,Xhat,Y,Yhat,params_sizes] = load_data2(b,K);
%     [~,~,X_test,Xhat_test,Y_test,Yhat_test,params_sizes_test] = load_data2(btest,K);


    PARA.n1 = params_sizes(1);
    PARA.n2 = params_sizes(2);
    PARA.n3 = params_sizes(3);
    PARA.n4 = params_sizes(4);
    PARA.N = params_sizes(5);

% run experiment
% 
% resTraining = [];
% resTesting = [];

% for K = K_exp
    
    PARA.K = K;
    PARAtest.K = K;
    
    t1 = tic;
    [~,Dhat,R_D] = tensor_trainer(Dhat,Xhat,Yhat,PARA);
    td = toc(t1);    
    fprintf('\nDone training K: %i! --> Time: %2.2f s\n\n', K, td)
    
    t2 = tic;
    if cellTest
        R_Z.NNZ_i = [];
        R_Z.CR_i = [];
        R_Z.PSNR_i = [];
        for i = 1:length(btest)
            [~,~,X_test,Xhat_test,Y_test,Yhat_test,params_sizes_test] = load_data2(reshape(btest{i},[size(btest{i}),1,1]),K);

            PARAtest.n1 = params_sizes_test(1);
            PARAtest.n2 = params_sizes_test(2);
            PARAtest.n3 = params_sizes_test(3);
            PARAtest.n4 = params_sizes_test(4);
            PARAtest.N = params_sizes_test(5);
            
            % Reshape dictionary
            D = ifft2(permute(Dhat,[3,4,2,1]),PARA.filter_szx,PARA.filter_szy);
            Dhat = permute(fft2(D,PARAtest.n3,PARAtest.n4),[4,3,1,2]);
            
            ti0 = tic;
            [~,~,R_Z_i] = tensor_trainer(Dhat,Xhat_test,Yhat_test,PARAtest);
            
            R_Z.NNZ_i = [R_Z.NNZ_i R_Z_i.NNZ];
            R_Z.CR_i = [R_Z.CR_i R_Z_i.CR];
            R_Z.PSNR_i = [R_Z.PSNR_i R_Z_i.PSNR];
            
            ti = toc(ti0);
            fprintf('\nDone testing signal: %i/%i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n',i,length(btest), ti,R_Z_i.PSNR,R_Z_i.CR)
        end
        
        R_Z.NNZ = sum(R_Z.NNZ_i);
        R_Z.CR = mean(R_Z.CR_i);
        R_Z.PSNR = mean(R_Z.PSNR_i);
        
    else
        [~,~,X_test,Xhat_test,Y_test,Yhat_test,params_sizes_test] = load_data2(btest,K);
        [~,~,R_Z] = tensor_trainer(Dhat,Xhat_test,Yhat_test,PARAtest);
    end
    
    
    tc = toc(t2);    
    fprintf('\nDone testing K: %i! --> Time: %2.2f s, PSNR: %.2f, CR: %.2f\n\n\n', K, tc,R_Z.PSNR,R_Z.CR)
    
    R_D.K = K;
    R_Z.K = K;
    
    resTraining = [resTraining, R_D];
    resTesting =  [resTesting, R_Z];

end


%% save

dataPath = [project,'/data/'];

try
    save2([dataPath,name,'TrainResults.mat'],'resTraining')
    save2([dataPath,name,'TestResults.mat'],'resTesting')
catch ME
    if strcmp(ME.identifier,'MATLAB:save:couldNotWriteFile')
        save2([dataPath,name,'TrainResults.mat'],'resTraining','resTraining','-noappend')
        save2([dataPath,name,'TestResults.mat'],'resTesting','resTraining','-noappend')
    end
end

dbclear all
