function TRcomp = TRComplexity(tensor)
dim = size(tensor);
term1 = (sum(1./sqrt(dim))/length(dim));
Fac = tucker(tensor, dim);
term2 = 0;
for i = 1:length(dim)
    term2 = term2 + sqrt(rank(Fac{i}))/length(dim);
end

TRcomp = (term1*term2)^2;
