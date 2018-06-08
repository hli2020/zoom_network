function show_box_simple( im, boxes, color )
%SHOW_BOX_SIMPLE Summary of this function goes here
%   Detailed explanation goes here

linewidth = 2;
if nargin < 3
    color = 'r';
end

hold on;
for i = 1:size(boxes, 1)
    box = boxes(i, :);
    rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor', color);
end
hold off;

function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
%rects (l, t, r, b) to (l, t, w, h)
rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end
end