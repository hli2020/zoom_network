function entry = change_edgebox( entry )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

entry(:, 3) = entry(:, 1) + entry(:, 3) - 1;
entry(:, 4) = entry(:, 2) + entry(:, 4) - 1;

end

