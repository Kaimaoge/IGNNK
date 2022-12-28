function [x, lam] = approxEV(A, ep)
MaxIter = 1000;
N = size(A, 1);
x = ones(N, 1);
lamp = (A*x)'*x/(norm(x)^2);
for i = 1:MaxIter
    x = A*x;
    x = x./max(abs(x));
    lam = (A*x)'*x/(norm(x)^2);
    
    if abs(lam - lamp) < ep
        if lam < 0
            [x, lam] = approxEV(A - (1.1*lam)*eye(N), ep);
        end
        x = x/norm(x);
        break
    else
        lamp = lam;
    end
end