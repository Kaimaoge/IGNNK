function err = normpredict(Y, X, Sol)
[~, ~, r] = size(Sol);
err = [0, 0];
for ll = 1:r
    err(1) = err(1) + norm(Y{ll} - squeeze(Sol(:, :, ll))*X{ll}, 'fro')^2;
    err(2) = err(1)/mean(Y{ll}(:).^2);
end
err(1) = err(1)/size(Y{1}, 2);
err = sqrt(err/r);
end