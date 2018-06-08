function res = stats_opt(x, y, mode)

switch mode
    case 'norm'
        res = [x(1)/y(2) x(2)/y(1) x(3)/y(2) x(4)/y(1)];
end
end