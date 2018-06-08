function rolling_mean( data, radius )
% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2015-2017 Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------

if nargin < 2
    radius = 50;
end

new_data = zeros(size(data));

for i = 1:length(data)
    start_ind = max(i-radius, 1);
    end_ind = min(i+radius, length(data));
    new_data(i) = mean(data(start_ind:end_ind));
end

figure;
grid on;
hold on;
plot(data);
plot(new_data, 'linewidth', 2);
hold off;
