function sim = haverSimple(locs, sigma)
% Input in the format of (n x 2) (lat, long) format
% Output in the form of (n x n)
n = size(locs, 1);
sim = zeros(n);

for i = 1:n
    for j = 1:n
        sim(i, j) = haversine(locs(i, :), locs(j, :));
    end
end

sim = sim - mean(mean(sim));
sim = sim/(std(sim(:), 0, 1));

sim = exp(-sim/sigma); % avoid overflow
sim = sim /sim(1, 1);

sim = diag(sum(sim)) - sim;  % Outputting  the laplacian

end