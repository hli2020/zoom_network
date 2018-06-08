function [rois, labels, bbox_targets, bbox_weights, conf, src_rois] = ...
    zoom_sample_rois(conf, im_size, gt_input, im_file_name, kk)
%
% An very important function to make samples
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%   Descends from 'sample_rois' in 'proposal_get_minibatch.m'
%
%   Output
%                 rois: for RoI
%               labels: [anchor_num x 1]
%         bbox_targets: [anchor_num x 4]
%             src_rois: all pos, resized RoI. WE JUST FORWARD POS ROIS

im_size_resize = im_size.resize;
im_size_ori = im_size.origin;
% only contains gt boxes, boxes in the original image
gt_rois = gt_input.boxes;
gt_labels = gt_input.class;
gt_rois_resize = scale_rois(gt_rois, im_size_ori, im_size_resize);

conf.temp_curr_level    = kk;
add_gray_cls            = conf.add_gray_cls;

%% locate/generate anchors
if conf.roi_followup
    % much fewer, say [489 x 4], or more, act as 'rois'
    anchors = double(scale_rois(gt_input.rpn_boxes, im_size_ori, im_size_resize));
    % VITAL UPDATE: WE DONT USE ALL ROIS
    % all_rois_rescale = anchors;
else
    anchors = proposal_locate_anchors(conf, im_size_resize);
end

% compute regression target on the fly
% 'bbox_targets_raw': [anchor_num x 5], 
%                       with first col has original labels (-2, -1, 0, 1, 2, ...)
% Output: bbox_targets
%           positive:   [class_label,   regression_label]   -> 1
%           ignore:     [-2,            -2, -2, -2, -2]     -> -1
%           negative:   [-1,            -2, -2, -2, -2]     -> 0
%           gray:       [0,             -2, -2, -2, -2]     -> 2
[bbox_targets_raw, src_rois, gt_info, new_anchors] = ...
    compute_targets(conf, gt_rois_resize, gt_labels, anchors, im_size_resize);

investigate.anchors = new_anchors;
investigate.gt_box_resize = gt_rois_resize;
conf.curr_level_gt_info = gt_info;

%% select fg/bg and set labels
rois_per_image = conf.batch_size;
fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
[fg_inds, bg_inds, choose_fg_index] = assign_pos_neg_sample(...
    bbox_targets_raw, fg_rois_per_image, rois_per_image, conf.adjust_sample_ratio);
% make labels: different from that in 'bbox_targets_raw' (mismatch for legacy reasons)
labels = -ones(size(bbox_targets_raw, 1), 1);
labels(bg_inds) = 0;
labels(fg_inds) = 1;

gray_cls_ind = [];
if add_gray_cls
    
    potential_gray_cls_ind = find(bbox_targets_raw(:, 1)==0);
    fg_bg_total_num = sum(labels==0) + sum(labels==1);  
    if fg_bg_total_num == 0
        gray_cls_num = min(floor(rois_per_image/8), length(potential_gray_cls_ind));
    else
        gray_cls_num = min(floor(fg_bg_total_num/2), length(potential_gray_cls_ind));
    end
    gray_cls_ind = potential_gray_cls_ind(randperm(length(potential_gray_cls_ind), gray_cls_num));
    assert(all(labels(gray_cls_ind) == -1));
    labels(gray_cls_ind) = 2;    
end

% NOTE: it's possible that this image has neither fg nor bg.
% chill up and don't worry. (accu=NaN, loss_cls=0, loss_bbox=0)
% if isempty(bg_inds) && isempty(fg_inds), keyboard; end
rois = [];
if conf.roi_followup
    keep_inds = [fg_inds; bg_inds; gray_cls_ind];
    labels = labels(keep_inds);
    rois = anchors(keep_inds, :);
end

%% generate bbox_targets
if ~conf.roi_followup
    
    bbox_targets = single(full(bbox_targets_raw(:, 2:end)));
    bbox_weights = bbox_targets * 0;
    bbox_weights(fg_inds, :) = 1;
else
    cls_num = 2;
    if add_gray_cls, cls_num = 3; end
    bbox_targets = zeros(length(keep_inds), 4*cls_num, 'single');
    bbox_weights = zeros(size(bbox_targets), 'single');
    
    for i = 1:length(fg_inds)
        bbox_targets(i, 5:8) = bbox_targets_raw(fg_inds(i), 2:end);
        bbox_weights(i, 5:8) = 1;
    end
end

%% debug: show image with anchors
if size(src_rois, 1) ~= 0, conf.temp_has_foreground(conf.temp_curr_level) = 1; end
% for imagenet
% whether_draw = conf.draw_boxes && ( ...
%     any(gt_label==91) || any(gt_label==12) || any(gt_label==69) || any(gt_label==52));
whether_draw = conf.draw_boxes;

if whether_draw
    src_rois_label = bbox_targets_raw(bbox_targets_raw(:, 1) > 0, 1);
    draw_damn_image(conf, im_file_name, gt_info, im_size_resize, ...
        src_rois, gt_rois, add_gray_cls, labels, ...
        fg_rois_per_image, rois_per_image, src_rois_label, choose_fg_index, investigate);
end
end

function [bbox_targets, src_rois, gt_info, new_anchors] = compute_targets( ...
    conf, gt_rois, gt_labels, anchors, im_size_resize)
%
% Descends from 'rpn_compute_targets_v0.m'
% NOTE:
%       gt_rois and anchors are all the resized (say 500 -> 600) results;
%       only 'bbox_targets' are needed output variable

if isempty(gt_rois)
    bbox_targets = zeros(size(anchors, 1), 5, 'double');
    bbox_targets(:, 1) = -1;
    return;
end

gt_labels = single(gt_labels);
assert(all(gt_labels > 0));

if conf.revise_boundary_box
    % revise boundary boxes, just keep them!
    [anchors_revise, check_] = refine_box(anchors, im_size_resize);
    % update anchors also!
    anchors(check_, :) = anchors_revise;
    new_anchors = anchors;
else
    error('revise_boundary_box NOT true');
end

% ov = compute_overlap(new_anchors, gt_rois);
% anchors_max_ov = extractfield(ov, 'max')';            % num_anchor x 1
% % each ranging from [0, gt_num] with 0 meaning no overlap to all GTs
% anchors_map_to_gt = extractfield(ov, 'max_ind')';     % num_anchor x 1
ov = boxoverlap(new_anchors, gt_rois);
[anchors_max_ov, anchors_map_to_gt] = max(ov, [], 2);
anchors_map_to_gt(anchors_max_ov==0) = 0;

% decide fg/bg
fg_inds = find( anchors_max_ov >= conf.fg_thresh );
bg_inds = anchors_max_ov < conf.bg_thresh_hi & anchors_max_ov >= conf.bg_thresh_lo ;
gray_inds = anchors_max_ov <= conf.gray_hi & anchors_max_ov >= conf.gray_lo ;

% only successfully recalled GTs are extracted
target_rois = gt_rois(anchors_map_to_gt(fg_inds), :);
src_rois = anchors(fg_inds, :);

gt_info_catch_ind = zeros(size(gt_labels, 1), 1);
gt_info_catch_ind(unique(anchors_map_to_gt(fg_inds))) = 1;
% col_vec | col_vec | col_vec
gt_info = [gt_info_catch_ind, single(gt_labels), gt_rois];

% we predict regression_label which is generated by an unlinear
% transformation from src_rois and target_rois
[regression_label] = fast_rcnn_bbox_transform(src_rois, target_rois);
% normalize target delta value
if ~isempty(fg_inds)
    try
        means = conf.bbox_means; stds = conf.bbox_stds;
    catch
        % if not specified, it remains unchanged
        means = zeros(1, 4); stds = ones(1, 4);
    end
    regression_label = bsxfun(@minus, regression_label, means);
    regression_label = bsxfun(@rdivide, regression_label, stds);
end

bbox_targets = -2*ones(size(anchors, 1), 5, 'double');
bbox_targets(fg_inds, :) = [gt_labels(anchors_map_to_gt(fg_inds)), regression_label];
bbox_targets(bg_inds, 1) = -1;
bbox_targets(gray_inds, 1) = 0;

end

function [anchors, check] = refine_box(anchors, im_size)
anchors(anchors(:, 1) <= 0, 1) = 1;
anchors(anchors(:, 2) <= 0, 2) = 1;

anchors(anchors(:, 3) > im_size(2), 3) = im_size(2);
anchors(anchors(:, 4) > im_size(1), 4) = im_size(1);

check = all( (anchors(:, 1) < anchors(:, 3) & anchors(:, 2) < anchors(:, 4)), 2 );
if sum(check) ~= length(anchors)
    warning('coordinates of some anchors mistaken');
end
anchors = anchors(check, :);
end

function [fg_inds, bg_inds, choose_fg_index] = assign_pos_neg_sample(...
    bbox_targets, fg_rois_per_image, rois_per_image, adjust_sample_ratio)

% Select foreground ROIs as those with >= FG_THRESH overlap
fg_inds = find(bbox_targets(:, 1) > 0);
% Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
bg_inds = find(bbox_targets(:, 1) == -1);

% select final foreground number and index
fg_num = min(fg_rois_per_image, length(fg_inds));
try
    choose_fg_index = randperm(length(fg_inds), fg_num);
    fg_inds = fg_inds(choose_fg_index);
catch
    choose_fg_index = []; fg_inds = [];
end
% select final background number and index
bg_num_pre = min(rois_per_image - fg_num, length(bg_inds));
if ~adjust_sample_ratio
    % select background as before
    bg_num = bg_num_pre;
else
    % adjust sample ratio
    if isempty(fg_inds)
        bg_num = min(rois_per_image/4, bg_num_pre);
    else
        bg_num = min(2*fg_num, bg_num_pre);
    end
end
try
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));
catch
    bg_inds = [];
end
end

