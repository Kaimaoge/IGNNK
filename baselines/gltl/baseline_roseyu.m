clc
clear
addpath(genpath('./'));
global verbose % Turns on the detailed output.  Set verbose to 0 to only 
verbose = 1;   % see the final results.   

%% Load the series data
% load metr_la_rdata.mat
% load unknow_set.mat

load nrel_rdata.mat
load nrel_unknow_set.mat
X = X(:,25920:end); % Too large to be full
    
% load udata_rdata.mat
% load udata_unknow_set.mat
% X = X/100; % Converted into meters of udata


% Reshape the input data, and only consider the first feature
% X = X(:,1,:); % For metr-la
% X = reshape(X,[size(X,1),size(X,3)]); % For metr-la
[P,T] = size(X);

% Split the training and test set along the time domain
split_line = int32(size(X,2) * 0.7);
training_set_mask = ones(size(X));
test_set_mask = ones(size(X));
training_set_mask(:,split_line:end) = 0;

% Construct masks for the nodes
full_set = 0:136; % 0:1217 For udata; % 0:136 % For nrel; % 0:206; % For metr-la
know_set = setdiff(full_set,unknow_set);

training_set_s_mask = zeros(size(training_set_mask));
training_set_s_mask(know_set+1,:) = 1; % Known graph mask

% Randomly choose 150/100/900 node for metr-la/nrel is observable
know_mask = randsample(know_set,100);
missing_node_mask = setdiff(know_set,know_mask); % For missing nodes masks

inputs = X;
inputs_omask = ones(size(inputs));
% inputs_omask(inputs == 0)=0; % COMMENT THIS FOR TEST ERROR ON MISSING DATA
inputs_omask(missing_node_mask,:) = 0;
% inputs_omask(unknow_set+1,:)=0; % COMMENT THIS FOR TEST ERROR ON IMPUTATION ERROR

mask_train = inputs_omask & training_set_mask & inputs;
mask_test = test_set_mask & inputs;
% mask_test(inputs == 0)=0; % COMMENT THIS FOR TEST ERROR ON MISSING DATA
mask_val = training_set_s_mask & inputs;
mask_val(unknow_set+1,:)=0;
% mask_val(inputs == 0)=0; % COMMENT THIS FOR TEST ERROR ON MISSING DATA
%% Construct the Laplacian matrix
% For metr-la
% location = readtable('metr_la_latlon.csv');
% location_vals = [location.longitude,location.latitude];
% sigma = 0.5; % Laplacian kernel parameter
% sim = haverSimple(location_vals, sigma);
% par.sim = sim/(max(sim(:))); 

% For nrel 
location_vals = [transpose(longitude), transpose(latitude)];
sigma = 0.5; % Laplacian kernel parameter
sim = haverSimple(location_vals, sigma);
par.sim = sim/(max(sim(:))); 

% For udata
% location = readtable('udata_latlon.csv');
% location_vals = [location.Var2,location.Var1];
% sigma = 0.5; % Laplacian kernel parameter
% sim = haverSimple(location_vals, sigma);
% par.sim = sim/(max(sim(:))); 
%% Set parameters 
par.eta = 1e-10; % convergence stopping criteria
par.max_iter = 10; % maximum number of iteration
par.mu = 5; % parameter for Laplacian regularizer
par.train_len = floor(T*0.8);  % training length for forecasting
par.num_lag = 2; % VAR model lag number
% 'forward' calls "forward greedy algorithm"
% 'ortho' calls "orthogonal greedy algorithm"
par.func = 'forward';

%% Cokriging 
% Missing index, to be noticed that 1 means missing and 0 means observed
inputs_cell = mat2cell(inputs,P,T); % Must convert the input into cells
par.metric = 'K'; % Evaluation function: Kriging
[sol_cokriging, quality_cokriging,runtime ] =  greedy_cokriging(inputs_cell,diag(~mask_train), par);
fprintf('cokring rmse %d \n', quality_cokriging(end));  
%% Calculate the error using our standard
rmse_test = sqrt(sum(sum(((sol_cokriging'-X).*mask_test).^2))/sum(sum(mask_test)))
mae_test = sum(sum( abs((sol_cokriging'-X).*mask_test)  ))/sum(sum(mask_test))
x_p = X;
x_p(x_p==0) = 1*10^(-20);
mape_test = sum(sum( abs((sol_cokriging'- X)./x_p.*mask_test)  ))/sum(sum(mask_test))

rmse_val = sqrt(sum(sum(((sol_cokriging'-X).*mask_val).^2))/sum(sum(mask_val)))
mae_val = sum(sum( abs((sol_cokriging'-X).*mask_val)  ))/sum(sum(mask_val))
mape_val = sum(sum( abs((sol_cokriging'- X)./x_p.*mask_val)  ))/sum(sum(mask_val))


figure
plot(sol_cokriging(:,3))
hold on
plot(X(3,:))
hold off
