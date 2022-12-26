function rmse = testQualityF(Sol, U, X, Y)
% compute RMSE for the testing series with transformation U
% used for forecasting
rmse = 0;
for i = 1:length(X)
    mat = U\squeeze(Sol(:, :, i));
    rmse = rmse + norm(mat*X{i} - Y{i}, 'fro')^2;
end
rmse = sqrt(rmse/length(X)/numel(Y{1}));