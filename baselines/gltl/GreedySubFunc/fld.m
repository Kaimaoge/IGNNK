function out = fld(matrix, mode, dim)
[dim1, dim2] = size(matrix);
switch mode
    case 1
        % Here 'dim' specifies 'r'
        r = dim; q = dim1; p = dim2/r;
        out = zeros(q, p, r);
        for i = 1:r
            out(:, :, i) = matrix(:, p*(i-1)+1:p*i);
        end
    case 2
        % Here 'dim' specifies 'r'
        r = dim; p = dim1; q = dim2/r;
        out = zeros(q, p, r);
        for i = 1:r
            out(:, :, i) = matrix(:, q*(i-1)+1:q*i)';
        end
    case 3
        % Here 'dim' specifies 'q'
        r = dim1; q = dim; p = dim2/q;
        out = zeros(q, p, r);
        for i = 1:r
            out(:, :, i) = reshape(matrix(i, :), p, q)';
        end
    otherwise
        error('Mode cannot be bigger than 3.')
end