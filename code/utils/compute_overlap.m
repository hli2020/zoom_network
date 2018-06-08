function [true_overlap, OVERLAP] = compute_overlap(pred_bbox_set, GT_bbox_set)
% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2015-2017 Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------
% input
%       pred_bbox_set:      pre_num x 4 absotue coordinate values
%       GT_bbox_set:        GT_num x 4 absotue coordinate values
%
% output
%       true_overlap:   structure
%                           overalp: each with GT_num x 1 overlap values
%                           max:     max value to all GTs
%                           max_ind: index of GTs
%
% updated on Oct.9th, 2015; Mar.8th, 2017
% very slow. consider using 'boxoverlap.m'

OVERLAP = zeros(size(pred_bbox_set, 1), size(GT_bbox_set, 1));
true_overlap = struct('overlap', [], 'max', [], 'max_ind', []);
true_overlap = repmat(true_overlap, [size(pred_bbox_set, 1) 1]);

for mm = 1:size(pred_bbox_set, 1)
    pred_bbox = pred_bbox_set(mm, :);
    pred_area = (pred_bbox(4)-pred_bbox(2)+1)*(pred_bbox(3)-pred_bbox(1)+1);
    overlap = zeros(size(GT_bbox_set,1), 1);
    for i = 1:size(GT_bbox_set, 1)
        
        GT_bbox = GT_bbox_set(i, :);
        GT_area = (GT_bbox(4)-GT_bbox(2)+1)*(GT_bbox(3)-GT_bbox(1)+1);
        
        if pred_bbox(3) < GT_bbox(1) || pred_bbox(1) > GT_bbox(3)
            x_overlap = 0;
        else
            total_x = max(pred_bbox(3), GT_bbox(3)) - ...
                min(pred_bbox(1), GT_bbox(1)) + 1;
            x_overlap = total_x - abs(GT_bbox(3) - pred_bbox(3)) ...
                - abs(GT_bbox(1) - pred_bbox(1));
        end
        if pred_bbox(4) < GT_bbox(2) || pred_bbox(2) > GT_bbox(4)
            y_overlap = 0;
        else
            total_y = max(pred_bbox(4), GT_bbox(4)) - ...
                min(pred_bbox(2), GT_bbox(2)) + 1;
            y_overlap = total_y - abs(GT_bbox(4) - pred_bbox(4)) ...
                - abs(GT_bbox(2) - pred_bbox(2));
        end
        
        intersection = x_overlap * y_overlap;
        union = GT_area + pred_area - intersection;
        overlap(i) = intersection / union;
    end
    OVERLAP(mm, :) = overlap';
    true_overlap(mm).overlap = single(overlap);
    [max_value, max_ind] = max(overlap);
    if sum(overlap) == 0, max_ind = 0; end
    true_overlap(mm).max = single(max_value);
    true_overlap(mm).max_ind = max_ind;
end
