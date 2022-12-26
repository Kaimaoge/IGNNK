function [delta, Sol] = solveFold2(Y, X, Sol)
% The goal of the following functions is to (1) Find the optimal rank-1
% direction in the given mode and (2) Report the amount of decrease in the
% objective function
[q, p, r] = size(Sol);
P = cell(r, 1);
Q = P;

% Create the Q, P matrices
for ll = 1:r
    P{ll} = X{ll}*X{ll}';

    Q{ll} = X{ll}*(Y{ll}'*Y{ll})*(X{ll}');
end

% Computing the solution
Max_Iter = 100;
obj = zeros(Max_Iter, 1);
u = ones(p, 1);
step = 1e-4;
for i = 1:Max_Iter
    [obj(i), G] = findGrad2(Q, P, u);
    u = u + step * (G);    % Added for regularization
end

v = zeros(q*r, 1);
for ll = 1:r
    v(q*(ll-1)+1:q*ll) = (Y{ll}*X{ll}'*u)/(u'*P{ll}*u);
end
SS = u*v';
Sol = fld(SS, 2, r);

% Computing delta
delta = 0;
for ll = 1:r
    delta = delta + norm(Y{ll}, 'fro')^2 - norm(Y{ll} - squeeze(Sol(:, :, ll))*X{ll}, 'fro')^2;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [obj, grad] = findGrad2(Q, P, v)
obj = 0;
grad = 0*v;
for i = 1:length(Q)
    temp = v'*Q{i}*v/(v'*P{i}*v);
    obj = obj + temp;
    grad = grad + 2*(Q{i}-P{i}*temp)*v/(v'*P{i}*v);
end
end