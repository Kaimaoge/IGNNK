function [Sol, quality] = ortho(Y, X, helper, max_iter, eta, ground_truth, metric)
% orthogonal greedy low rank tensor learing  algorithm
% X and Y are cells of size nTask
% Y{i} is a matrix of size (nPred) x (nData)
% X{i} is a matrix of size (nFeature) x (nData)

global verbose
r = length(X);
[p, n] = size(X{1});
q = size(Y{1}, 1);

Sol = zeros(q, p, r);
tempSol = cell(3, 1);
delta = zeros(3, 1);
obj = zeros(max_iter, 1);
quality = zeros(max_iter, 5);

Yp = Y;

if verbose; fprintf('Iter #: %5d', 0); end
for ll = 1:r; obj(1) = obj(1) + norm(Y{ll}, 'fro')^2; end
for i = 1:max_iter-1
    [delta(1), tempSol{1}] = solveFold1(Yp, X, Sol);
    [delta(2), tempSol{2}] = solveFold2(Yp, X, Sol);
    [delta(3), tempSol{3}] = solveFold3(Yp, X, Sol);
    [~, ix] = max(delta);
   if delta(ix)/obj(1) > eta
        Sol = Sol + tempSol{ix};
        [Yp, Sol, obj(i+1)] = project(Y, X, Sol, i); % Do an orthogonal projection step here
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

if verbose; fprintf('\n'); end
quality = [obj, quality];
quality(i+1:end, :) = [];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y, Sol, obj] = project(Y, X, Sol, order)
% I choose to always make the projection in the first mode
[q, p, r] = size(Sol);
n = size(X{1}, 2);
matrix = unfld(Sol, 1);

% Setting up the matrices such that Loss = |YY - A x XX|_F^2
XX = zeros(p*r, n*r);
YY = zeros(q, n*r);
for ll = 1:r
    XX( p*(ll-1)+1:p*ll , n*(ll-1)+1:n*ll ) = X{ll};
    YY( : , n*(ll-1)+1:n*ll ) = Y{ll};
end

% Finding the bases
if order > min(size(matrix))
    order = min(size(matrix));
    [U, ~, V] = svds(matrix, order);
else
    [U, ~, V] = svds(matrix, order);
end

XXX = V'*XX;
YYY = U'*YY;
B = (YYY*XXX')/(XXX*XXX');
Sol = fld(U*B*V', 1, r);
obj = 0;
for ll = 1:r; 
    Y{ll} = Y{ll} - squeeze(Sol(:, :, ll))*X{ll}; 
    obj = obj + norm(Y{ll}, 'fro')^2;
end
end
