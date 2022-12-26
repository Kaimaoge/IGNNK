function [sol,quality,runTime] = greedy_forecasting(data_series, par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% greedy low rank tensor learning for forecasting task
% See ``fast multivariate spatio-temporal analysis via low rank tensor
%       learning [NIPS 2014] '' for details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data_series: M x {P x Q } cell array,  complete data series
% par:  Parameters
%       par.train_len: constant, length of the series used for training
%       par.num_lag: constant, VAR model lag
%       par.mu: constant, Laplacian constraint parameter
%       par.sim: P x P matrix, similarity matrix 
%       par.max_iter: constant, maximum number of iteration
%       par.eta: constant, stopping criteria
%       par.metric: string, evaluation metric
%           -'F': testing RMSE
%           -'R': testing RMSE evaluated at non-zero entries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sol: P x PK x M matrix,  solution tensor
% quality: num_iter x 1 vector, reporting evaluation quality at each
%          iteration
% runTime: Runtime of the algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_len = par.train_len;
num_lag = par.num_lag;

M = length(data_series);
[P,T ] = size(data_series{1});

test_series.X = cell(M, 1);
test_series.Y = cell(M, 1);

X = cell(M, 1);
Y = cell(M, 1);

U = chol(eye(P) + par.mu *par.sim);
for i = 1:M
    Y{i} = data_series{i}(:, num_lag+1:train_len);
    Y{i} = (U')\Y{i};               % This is the transformation
    test_series.Y{i} = data_series{i}(:, train_len-num_lag+1:T);
    X{i} = zeros(num_lag*P, (train_len - num_lag));
    for ll = 1:num_lag
        X{i}(P*(ll-1)+1:P*ll, :) = data_series{i}(:, num_lag+1-ll:train_len-ll);
        test_series.X{i}(P*(ll-1)+1:P*ll, :) = data_series{i}(:, train_len-num_lag+1-ll:T-ll);
    end
end

tic
[sol, quality] = feval(par.func, Y, X, U,  par.max_iter,  par.eta, test_series, par.metric);
runTime = toc;

end

