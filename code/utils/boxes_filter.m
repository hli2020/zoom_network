function [aboxes_per_im_out, ind] = boxes_filter(aboxes_per_im_in, ...
    nms_overlap_thres, after_nms_topN, per_nms_topN)

if nargin < 4, per_nms_topN = -1; end

% per_nms_topN = -1
% nms_overlap_thres = 0.7
% after_nms_topN = 2000
%
% update: 'ind' outputs the index from 'aboxes_per_im_in'
% each entry inside aboxes is a single matrix, say [12300 x 5]

if ~iscell(aboxes_per_im_in)
    aboxes = cell(1, 1); aboxes{1} = aboxes_per_im_in;
else
    aboxes = aboxes_per_im_in;
end
ind = cell(size(aboxes));

%%
if per_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), ...
        aboxes, 'UniformOutput', false);
end

use_gpu = true;
assert(nms_overlap_thres > 0 && nms_overlap_thres < 1);
for i = 1:length(aboxes)
    ind_temp = nms(aboxes{i}, nms_overlap_thres, use_gpu);
    aboxes{i} = aboxes{i}(ind_temp, :);
    ind{i} = ind_temp(1 : min(size(ind_temp,1), after_nms_topN));
end

if after_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(size(x,1), after_nms_topN), :), ...
        aboxes, 'UniformOutput', false);
end

%%
if ~iscell(aboxes_per_im_in)
    aboxes_per_im_out = aboxes{1};
else
    aboxes_per_im_out = aboxes;
end
end


