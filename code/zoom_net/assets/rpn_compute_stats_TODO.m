function [bbox_means, bbox_stds] = ...
    proposal_prepare_image_roidb_on_the_fly(conf, imdbs, roidbs)
% --------------------------------------------------------
% formerly known as 'proposal_prepare_image_roidb_on_the_fly'
% now changed to 'rpn_compute_bbox_stats'
% --------------------------------------------------------

if ~iscell(imdbs)
    imdbs = {imdbs};
    roidbs = {roidbs};
end

imdbs = imdbs(:);
roidbs = roidbs(:);

assert(conf.target_only_gt==true);
roidb_merge = []; imdb_merge_path = []; imdb_merge_size = [];
for i = 1:length(imdbs)
    assert( strcmp(imdbs{i}.name, roidbs{i}.name) );        % name match
    assert(length(imdbs{i}.image_ids) == length(roidbs{i}.rois)); % number match
    roidb_merge = [roidb_merge; roidbs{i}.rois'];
    imdb_merge_path = [imdb_merge_path; imdbs{i}.image_ids];
    imdb_merge_size = [imdb_merge_size; imdbs{i}.sizes];
    im_path_root{i} = imdbs{i}.image_dir;
end
num_images = length(roidb_merge);
fprintf(' || RPN: see via (%d) training images...\n', num_images);
%%
fprintf(' || RPN: begin to compute bbox regression targets and stats...\n');
% Compute values needed for means and stds
% means and stds -- k * 4, include background class, here k = 1 for rpn
% var(x) = E(x^2) - E(x)^2
class_counts = zeros(1, 1) + eps;
sums = zeros(1, 4);
squared_sums = zeros(1, 4);

for i = 1:num_images
    im_size = imdb_merge_size(i, :);
    [anchors, im_scales] = proposal_locate_anchors(conf, im_size);
    gt_rois = roidb_merge(i).boxes;             % only contains gt boxes
    gt_labels = roidb_merge(i).class;
    curr_im_bbox_targets = cellfun(@(x, y) ...
        hyli_rpn_compute_targets(conf, scale_rois(gt_rois, im_size, y), gt_labels, x, im_size, y), ...
        anchors, im_scales, 'UniformOutput', false);
    for j = 1:length(conf.scales)
        targets = curr_im_bbox_targets{j};
        gt_inds = find(targets(:, 1) > 0);
        if ~isempty(gt_inds)
            class_counts = class_counts + length(gt_inds);
            sums = sums + sum(targets(gt_inds, 2:end), 1);
            squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
        end
    end
    
end
bbox_means = bsxfun(@rdivide, sums, class_counts);
bbox_stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), bbox_means.^2)).^0.5;

% % Normalize targets
% for i = 1:num_images
%     for j = 1:length(conf.scales)
%         targets = image_roidb(i).bbox_targets{j};
%         gt_inds = find(targets(:, 1) > 0);
%         if ~isempty(gt_inds)
%             image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
%                 bsxfun(@minus, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), means);
%             image_roidb(i).bbox_targets{j}(gt_inds, 2:end) = ...
%                 bsxfun(@rdivide, image_roidb(i).bbox_targets{j}(gt_inds, 2:end), stds);
%         end
%     end
% end

fprintf(' || done!\n');
end


function scaled_rois = scale_rois(rois, im_size, im_scale)
% add the following to prevent empty box in roidb
if isempty(rois)
    scaled_rois=[];
    return
end
im_size_scaled = round(im_size * im_scale);
scale = (im_size_scaled - 1) ./ (im_size - 1);
scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;

end

