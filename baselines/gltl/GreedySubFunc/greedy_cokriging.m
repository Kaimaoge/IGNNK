function [sol,quality,runTime] = greedy_cokriging(data_series, idx_missing, par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% greedy low rank tensor learning for cokriging task
% See ``fast multivariate spatio-temporal analysis via low rank tensor
%       learning [NIPS 2014] '' for details.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data_series: M x {P x Q } cell array,  complete data series
% idx_missing: P x 1 vector, indicating whether the location is missing(1)
%              or observed (0); 
% par:  The parameters structure:
%       par.mu: constant, Laplacian constraint parameter
%       par.sim: P x P matrix, similarity matrix 
%       par.max_iter: constant, maximum number of iteration
%       par.eta: constant, stopping criteria
%       par.func:  'forward' or 'ortho', specifying the greedy algorithm.
%       par.metric: string, evaluation metric
%        -'K': testing RMSE
%        -'R': testing RMSE evaluated at non-zero entries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sol: P x PK x M matrix,  solution tensor
% quality: num_iter x 1 vector, reporting evaluation quality at each
%          iteration
% runTime: Runtime of the algorithm.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = par.mu;
sim = par.sim;
test_idx = find(idx_missing==1);   
I_omega = diag(~idx_missing); % missing index should be 1 


M = length(data_series);
X = cell(M, 1);
Y = cell(M, 1);
for i = 1:M
    Q = chol(I_omega + mu * sim);
    M = (Q')\(I_omega*data_series{i});
    Y{i} = M';
    X{i} = Q';
end
tic
[sol, quality] = feval(par.func, Y, X, test_idx,  par.max_iter,  par.eta, data_series, par.metric);
runTime = toc;

end

