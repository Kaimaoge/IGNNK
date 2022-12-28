function out = unfld(tensor, mode)
[q, p, r] = size(tensor);
switch mode
    case 1
        out = zeros(q, p*r);
        for i = 1:r
            out(:, p*(i-1)+1:p*i) = squeeze(tensor(:, :, i));
        end
    case 2
        out = zeros(p, q*r);
        for i = 1:r
            out(:, q*(i-1)+1:q*i) = squeeze(tensor(:, :, i))';
        end
    case 3
        out = zeros(r, p*q);
        for i = 1:r
            out(i, :) = reshape(squeeze(tensor(:, :, i)), 1, p*q)'; % Is there any '?
        end
    otherwise
        error('Mode cannot be bigger than 3.')
end