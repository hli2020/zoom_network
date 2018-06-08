function draw_damn_image(conf, im_file_name, gt_info, im_size_resize, ...
    src_rois, gt_rois, add_gray_cls, labels, ...
    fg_rois_per_image, rois_per_image, src_rois_label, choose_fg_index, investigate)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

gt_label = gt_info(:, 2);
gt_all = gt_info(:, 3:end);
gt_all_hw = floor([gt_all(:, 4)-gt_all(:, 2), gt_all(:, 3)-gt_all(:, 1)]);
figure(conf.temp_curr_level);
im = imread(im_file_name);
im = bsxfun(@minus, single(im), conf.DATA.image_mean);
target_size = im_size_resize;
im_resize = imresize(im, target_size, 'bilinear', 'antialiasing', false);

hold on;
imshow(mat2gray(im_resize));
gt_info = double(round(gt_info));
gt_rois_success = gt_info(logical(gt_info(:,1)), 3:end);
gt_rois_fail = gt_info(~logical(gt_info(:,1)), 3:end);
% % temp
% ld = load('./data/datasets/ilsvrc14_det/ILSVRC2014_devkit/data/meta_det.mat');
% temp = ld.synsets_det(1:200);
% gt_str = extractfield(temp(gt_label), 'name')';
% gt_str_success = gt_str(logical(gt_info(:,1)));
if size(src_rois, 1) ~= 0
    
    %     cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
    %         num2cell(src_rois, 2));
    cellfun(@(x, y) text(x(1), x(2), sprintf('%d', y), 'backgroundcolor', 'g'), ...
        num2cell(src_rois, 2), num2cell(src_rois_label));
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g', 'LineWidth', 1), ...
        num2cell(src_rois(choose_fg_index, :), 2));
    % draw a single pos sample
    rectangle('Position', RectLTRB2LTWH(src_rois(1,:)), 'EdgeColor', 'b', 'linewidth', 2);
    text(src_rois(1,1), src_rois(1,2), sprintf('%d', src_rois_label(1)), 'color', 'w', ...
        'backgroundcolor', 'b');
end
% draw GT
cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r', 'linewidth', 2), ...
    num2cell(gt_rois_success, 2));
cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r', ...
    'linewidth', 2, 'linestyle', '--'), ...
    num2cell(gt_rois_fail, 2));
cellfun(@(x, y, z) text(x(3), x(2), sprintf('%d (%d,%d)', y, z(1), z(2)), 'backgroundcolor', 'r'), ...
    num2cell(double(gt_all), 2), num2cell(gt_label), num2cell(gt_all_hw, 2));

%============
anchors = investigate.anchors;
gt_rois_resize = investigate.gt_box_resize;
overlap = compute_overlap(gt_rois_resize, anchors);
nearest_anchor = anchors(extractfield(overlap, 'max_ind'), :);
ov_score = extractfield(overlap, 'max');
gt_indicator = ov_score>0;
w = size(im_resize, 2); h = size(im_resize, 1);
nearest_anchor(:, 1) = max(nearest_anchor(:, 1), 1);
nearest_anchor(:, 2) = max(nearest_anchor(:, 2), 1);
nearest_anchor(:, 3) = min(nearest_anchor(:, 3), w);
nearest_anchor(:, 4) = min(nearest_anchor(:, 4), h);

for fuck = 1:length(gt_indicator)
    if gt_indicator(fuck)
        rectangle('Position', RectLTRB2LTWH(nearest_anchor(fuck, :)), 'EdgeColor', 'y', 'linewidth', 1.5);
        text(nearest_anchor(fuck,3), nearest_anchor(fuck,4), sprintf('%.2f', ov_score(fuck)), 'color', 'b', ...
            'backgroundcolor', 'y');
    end
end
hold off;

% write batch number info and others
[~, im_name, ~] = fileparts(im_file_name);
title(sprintf('level: %d, feat stride: %d\nindex, %d, %s (resized image: %d x %d)\nall pos anchor boxes (%d, green), GT boxes (total %d in red, %d failed)', ...
    conf.temp_curr_level, conf.rpn_feat_stride(conf.temp_curr_level), ...
    conf.temp_which_ind, strrep(im_name, '_', '\_'), size(im_resize, 1), size(im_resize, 2), ...
    size(src_rois, 1), size(gt_rois, 1), sum(~gt_info(:,1))));

if ~add_gray_cls
    text(1, target_size(1), sprintf(...
        'actual fg/bg:: %d/%d\ndefault fg/batchSize:: %d/%d', ...
        length(find(labels>0)), length(find(labels==0)), ...
        fg_rois_per_image, rois_per_image), 'backgroundcolor', 'g');
else
    text(1, target_size(1), sprintf(...
        'actual fg/bg:: %d/%d\ndefault fg/batchSize:: %d/%d\ngray cls num:: %d', ...
        length(find(labels==1)), length(find(labels==0)), ...
        fg_rois_per_image, rois_per_image, ...
        length(find(labels==2))), 'backgroundcolor', 'g');
end
show_anchor_folder = sprintf('./output/visualize_anchor/%s', conf.model_id);
mkdir_if_missing(show_anchor_folder);

print(gcf, '-dpng', '-r0', sprintf('%s/ind_%d_level_%d.jpg', ...
    show_anchor_folder, conf.temp_which_ind, conf.temp_curr_level));
print(sprintf('%s/ind_%d_level_%d.pdf', ...
    show_anchor_folder, conf.temp_which_ind, conf.temp_curr_level), '-dpdf', '-r0');
end
