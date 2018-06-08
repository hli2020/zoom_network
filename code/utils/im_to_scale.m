function [im, bbox] = im_to_scale(im_ori, bbox_ori, ...
    shorter_dim_range, max_dim, small_case)
% the function below should be a generic method for all tasks
% NOTE: this function resides in the 'cvpr17_proposal_dev' repo

if nargin < 5
    use_smaller_dim = false; 
else
    use_smaller_dim = true;
end

if size(im_ori, 3) ~= 3
    im_ori = repmat(im_ori, [1 1 3]);
end

try
    shorter_dim = randi([shorter_dim_range(1) shorter_dim_range(2)], 1);
catch
    assert(length(shorter_dim_range)==1);
    shorter_dim = shorter_dim_range;
end

% (80% prob with the shorter_dim above) OR (20% prob with 'conf.small_dim')
if use_smaller_dim
    if rand(1) < 0.2
        shorter_dim = small_case.shorter_dim;
        max_dim = small_case.max_dim;
    end
end

h = size(im_ori, 1); w = size(im_ori, 2);
if h>= w
    scale = shorter_dim / w;
    new_h = floor(h * scale);
    im = imresize(im_ori, [new_h shorter_dim]);
else
    scale = shorter_dim / h;
    new_w = floor(w * scale);
    im = imresize(im_ori, [shorter_dim new_w]);
end
bbox = floor(bbox_ori.*scale);

% re-compute image if larger dim exceeds max_dim
larger_dim = max(size(im));
if larger_dim > max_dim
    scale_down_factor = max_dim/larger_dim;
    im = imresize(im, scale_down_factor);
    bbox = floor(bbox.*scale_down_factor);
end

new_w = size(im, 2);
new_h = size(im, 1);
% check bbox coordinate
bbox_num = size(bbox, 1);
bbox(:, 1) = max(bbox(:, 1), ones(bbox_num, 1)); 
bbox(:, 2) = max(bbox(:, 2), ones(bbox_num, 1));
bbox(:, 3) = min(bbox(:, 3), repmat(new_w, [bbox_num, 1]));
bbox(:, 4) = min(bbox(:, 4), repmat(new_h, [bbox_num, 1]));