function [delta, Sol] = solveFold3(Y, X, Sol)
% The goal of the following functions is to (1) Find the optimal rank-1
% direction in the given mode and (2) Report the amount of decrease in the
% objective function
[q, p, r] = size(Sol);
Max_Iter = 100;
obj = zeros(Max_Iter, 1);
A = ones(q, p);
step = 1e-3;

Y = Y{1};
X = X{1};

XYT = X*Y'; 
XXT = X*X';
for i = 1:Max_Iter
    [obj(i), G] = findGrad3(Y, X, A, XYT, XXT);
    A = A + step * G;
end

u = zeros(r, 1);
for i = 1:r
    u(i) = trace(A*X*Y')/norm(A*X, 'fro')^2;
end

SS = u*reshape(A', 1, p*q); %%%
Sol = fld(SS, 3, q);

% Computing delta
delta = 0;
for ll = 1:r
    delta = delta + norm(Y, 'fro')^2 - norm(Y - squeeze(Sol(:, :, ll))*X, 'fro')^2;
end
end


function [obj, G] = findGrad3(Y, X, A, XYT, XXT )
obj = 0;
G = 0*A;

m = norm(A*X, 'fro')^2;
[sa,sb] = size(A);
if sa<sb
    tr = trace(A*XYT);
else
    tr = trace(XYT*A);
end
obj = obj + tr^2/m;
%mp = 2*tr*(XYT');
G = G + 2*tr*(XYT')/m - (2*(tr^2)/(m^2))*(A*XXT);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [obj, G] = findGrad3( Y, X, A )
% obj = 0;
% G = 0*A;
% for i = 1:length(Y)
%     m = norm(A*X{i}, 'fro')^2;
%     tr = trace(A*X{i}*Y{i}');
%     obj = obj + tr^2/m;
%     mp = 2*tr*(Y{i}*X{i}');
%     G = G + mp/m - 2*A*(X{i}*X{i}')*(tr^2)/(m^2);
% end
% end