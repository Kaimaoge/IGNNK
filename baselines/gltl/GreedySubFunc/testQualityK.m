function rmse =  testQualityK(estMat, index, trueMat)
% compute the RMSE of test samples index ( missing entries )
% used for Kriging
rmse = 0;
for i = 1:length(trueMat)
    rmse = rmse + norm(squeeze(estMat(:, index, i))' - trueMat{i}(index, :), 'fro')^2;
end
rmse = sqrt(rmse/length(trueMat)/size(trueMat{1}, 2)/length(index));


end

