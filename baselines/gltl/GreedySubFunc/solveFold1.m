function [delta, Sol] = solveFold1(Y, X, Sol)
% The goal of the following functions is to (1) Find the optimal rank-1
% direction in the given mode and (2) Report the amount of decrease in the
% objective function
[q, p, r] = size(Sol);
n = size(X{1}, 2);
lam = 1e-3;
XX = zeros(p*r, n*r);
YY = zeros(q, n*r);
for ll = 1:r
    XX( p*(ll-1)+1:p*ll , n*(ll-1)+1:n*ll ) = X{ll};
    YY( : , n*(ll-1)+1:n*ll ) = Y{ll};
end
Q = XX*YY';
Q = Q*Q';
P = XX*XX';

% My solution
[~, lamU] = approxEV(YY*YY', 1e-4);  % our fast approximation to the generalized eigenvalue problem
[v, ~] = approxEV(Q-lamU*P, 1e-4);
% Matlab's solution
% [~, lamU] = eigs(YY*YY', 1);
% [v, ~] = eigs(Q-lamU*P, 1);

u = (YY*XX'*v)/(v'*P*v);
SS = u*v';
Sol = fld(SS, 1, r);

delta = 0;
for ll = 1:r
    delta = delta + norm(Y{ll}, 'fro')^2 - norm(Y{ll} - squeeze(Sol(:, :, ll))*X{ll}, 'fro')^2;
end
end