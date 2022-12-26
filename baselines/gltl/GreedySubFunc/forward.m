function [Sol, quality] = forward(Y, X, helper, max_iter, eta, ground_truth, metric)
% forward greedy low rank tensor learing  algorithm
global verbose
r = length(X);
[p, n] = size(X{1});
q = size(Y{1}, 1);

Sol = zeros(q, p, r);
tempSol = cell(3, 1);
delta = zeros(3, 1);
obj = zeros(max_iter, 1);
quality = zeros(max_iter, 1);

for ll = 1:r; obj(1) = obj(1) + norm(Y{ll}, 'fro')^2; end
if verbose; fprintf('Iter #: %5d', 0); end
for i = 1:max_iter-1
    [delta(1), tempSol{1}] = solveFold1(Y, X, Sol);
    [delta(2), tempSol{2}] = solveFold2(Y, X, Sol);
    [delta(3), tempSol{3}] = solveFold3(Y, X, Sol);
    [~, ix] = max(delta);
   if delta(ix)/obj(1) > eta
        Sol = Sol + tempSol{ix};
        for ll = 1:r; Y{ll} = Y{ll} - squeeze(tempSol{ix}(:, :, ll))*X{ll}; end
            obj(i+1) = obj(i) - delta(ix);
   else 
        break
   end
    
   if strcmp(metric,'K')
    % RMSE for cokriging
    quality(i+1) = testQualityK(Sol, helper, ground_truth);
   elseif strcmp(metric, 'R')
    % RMSE for cokriging rating, ignore zero entries
    quality(i+1) = testQualityR(Sol, helper, ground_truth);
   elseif strcmp(metric, 'F')
    % RMSE for forecasting
    quality(i+1) = testQualityF(Sol, helper, ground_truth.X, ground_truth.Y);
   end
    if verbose
        fprintf('%c%c%c%c%c%c', 8,8,8,8,8,8);
        fprintf('%5d ', i);
    end
end

quality(i+1:end) = [];
% plot(1:i, obj(1:i))
if verbose;
    fprintf('\n'); 
end
end