function [input_blobs, gt_info, gt_stats, all_rois_blob, im_size] = ...
    proposal_get_minibatch(conf, imdb_info, curr_recursive_rpn_box)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%
% currently only allow for num_images = 1
% for each image in each gpu
% --------------------------------------------------------
% ====================
% imdb_info.imdb_merge_path{1} = 'ILSVRC2012_val_00004246';
% imdb_info.imdb_merge_size = [400, 500];
% imdb_info.roidb_merge.boxes = [145,34,331,326;2,266,500,400];
% imdb_info.roidb_merge.class = [124;177];
% ld = load('./ILSVRC2012_val_00004246.mat');
% temp = ld.aboxes;
% imdb_info.roidb_merge.rpn_boxes = temp(:, 1:4);
% clear temp; clear ld;
% ====================

if conf.roi_followup
    
    if conf.use_coco
        % coco case
        ld = load([conf.rpn_box_path('coco_train') '/' ...
            imdb_info.imdb_merge_path{1}(1:end-4) '.mat']);
    else
        % imagenet case
        try
            ld = load([conf.rpn_box_path('train14') '/' imdb_info.imdb_merge_path{1} '.mat']);
        catch
            ld = load([conf.rpn_box_path('val1') '/' imdb_info.imdb_merge_path{1} '.mat']);
        end
    end
    temp = ld.aboxes;   
    if isempty(conf.recursive_rpn_box)
        % init, T = 1
        imdb_info.roidb_merge.rpn_boxes = temp(:, 1:4);
    else
        % update the boxes based on previous prediction
        % T >= 2
        imdb_info.roidb_merge.rpn_boxes = [curr_recursive_rpn_box; temp(:, 1:4)];
    end
end
roidb_merge = imdb_info.roidb_merge;
imdb_merge_size = imdb_info.imdb_merge_size;
im_size.origin = imdb_merge_size;
try
    compute_stats = conf.compute_stats;
catch
    compute_stats = false;
end
% interesting: wont show under parfeval condition!
% disp('hello');

%% resize image
% change 'conf.scales' here (new shorter dim of image)
rpn_min_size = conf.rpn_min_size;
if conf.dynamic_train_scale
    % change the scale for this time
    gt_rois = roidb_merge.boxes;
    conf.scales = dynamic_change_scale(conf, gt_rois, imdb_merge_size);
end

if conf.scales < rpn_min_size   
    temp_scale = conf.scales; 
    conf.scales = rpn_min_size;
    fprintf('resized image scale too small:: %d, change to:: %d\n', ...
        temp_scale, conf.scales);
end

% Sample random scales to use for each image in this batch
random_scale_inds = randi(length(conf.scales), 1);  % scalar
% resize the image, consider max_size, and select from certain scale
[im_blob, im_file_name] = get_image_blob(conf, imdb_info, random_scale_inds);

%% prepare input blob
level_num = length(conf.anchors);
im_size_resize = size(im_blob);
level_sub_blob = cell(level_num, 1);
conf.temp_has_foreground = zeros(level_num, 1);
im_size.resize = im_size_resize(1:2);
% change anchor template dynamically
try
    if conf.use_new_anchor
        conf.anchors = template_map_to_real_anchor(conf.anchor_pattern, im_size_resize);
    end
catch
end

gt_info = [];
input_blobs = [];
for kk = 1:level_num
    
    % we must convey the level-msg to sub-function
    % 'rois' is the resized boxes
    [rois, labels, bbox_targets, bbox_weights, conf, all_rois_rescale] = ...
        zoom_sample_rois(conf, im_size, imdb_info.roidb_merge, im_file_name, kk);
    
    if ~compute_stats
        
        % reshape the data
        if conf.roi_followup
            
            rois_blob = [ones(size(rois, 1), 1), floor(rois)];
            rois_blob = rois_blob - 1; % to c's index (start from 0)
            rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
            
            labels_blob = single(permute(labels, [3, 4, 2, 1]));
            bbox_targets_blob = single(permute(bbox_targets, [3, 4, 2, 1]));
            bbox_weight_blob = single(permute(bbox_weights, [3, 4, 2, 1]));
            level_sub_blob{kk} = {rois_blob, labels_blob, bbox_targets_blob, bbox_weight_blob};
            
        else            
            output_size = cell2mat([conf.output_height_map{kk}.values({im_size_resize(1)}), ...
                conf.output_width_map{kk}.values({im_size_resize(2)})]);
            labels_blob = reshape(labels, size(conf.anchors{kk}, 1), output_size(1), output_size(2));
            bbox_targets_blob = reshape(bbox_targets', size(conf.anchors{kk}, 1)*4, output_size(1), output_size(2));
            bbox_weight_blob = reshape(bbox_weights', size(conf.anchors{kk}, 1)*4, output_size(1), output_size(2));       
            % permute from [channel, height, width, num]
            % to [width, height, channel, num]
            labels_blob = single(permute(labels_blob, [3, 2, 1, 4]));
            bbox_targets_blob = single(permute(bbox_targets_blob, [3, 2, 1, 4]));
            bbox_weight_blob = single(permute(bbox_weight_blob, [3, 2, 1, 4]));
            level_sub_blob{kk} = {labels_blob, bbox_targets_blob, bbox_weight_blob};
        end
        
    end
    if kk == 1
        gt_info = [conf.curr_level_gt_info(:, 2) conf.curr_level_gt_info(:, 1)];
    else
        gt_info = [gt_info conf.curr_level_gt_info(:, 1)];
    end
end
all_rois_blob = [ones(size(all_rois_rescale, 1), 1), floor(all_rois_rescale)];
all_rois_blob = all_rois_blob - 1; % to c's index (start from 0)
all_rois_blob = single(permute(all_rois_blob, [3, 4, 2, 1]));

gt_stats.gt_resize_rois = conf.curr_level_gt_info(:, 2:end);
gt_stats.im_size_resize = im_size_resize;

if conf.draw_boxes
    if any(conf.temp_has_foreground)
        conf.temp_has_foreground
        each_gt_catch = max(gt_info(:, 2:end), [], 2);
        if sum(each_gt_catch) == length(each_gt_catch)
            fprintf('all %d GTs retrieved\n', sum(each_gt_catch));
        else
            fprintf('%d GTs (tot: %d) NOT retrieved\n', ...
                length(each_gt_catch)-sum(each_gt_catch), ...
                sum(each_gt_catch));
        end
        keyboard;
    else
        fprintf('fuck! id: %d, all levels failed to detect pos samples\n', conf.temp_which_ind);
        keyboard;
    end
end

if ~compute_stats
    if conf.debug
        if ~any(conf.temp_has_foreground)
            try
                cprintf('blue', ...
                    'fuck! gpu_id #%d, all levels failed to detect pos samples\n', ...
                    conf.temp_which_gpu); % this var is assigned in 'rpn_fetch_data.m'
            catch
                cprintf('blue', ...
                    'fuck! gpu_id #%d, all levels failed to detect pos samples\n', 0);
            end
        end
    end
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    
    input_blobs{1} = im_blob;
    cnt = 0;
    for kk = 1:level_num
        sub_length = length(level_sub_blob{kk});
        for shit = 1:sub_length
            cnt = cnt + 1;
            input_blobs{1+cnt} = level_sub_blob{kk}{shit};
        end
    end
end
end

function [im_blob, im_name_complete] = get_image_blob(conf, imdb_info, random_scale_inds)

im_name_ext = '';
if strcmp(conf.dataset, 'imagenet'), im_name_ext = '.JPEG'; end
if strcmp(conf.dataset, 'voc'), im_name_ext = '.jpg'; end

% add here to change image path for each image on 'imagenet_3k' set.
% not sure if this will slow down the processing time
% process_prefix: n04409011_17998.JPEG
if strcmp(conf.dataset, 'imagenet_3k')
    process_prefix = [imdb_info.imdb_merge_path{1}(1:9) '/' imdb_info.imdb_merge_path{1}];
else
    process_prefix = imdb_info.imdb_merge_path{1};
end

conf.server_at_bj = false; % flip the image on-the-fly (deprecated)
check_ = false;
if conf.server_at_bj && strcmp(process_prefix(end-3:end), 'flip')
    % remove '_flip' sign since we dont have such images on bj server
    process_prefix = process_prefix(1:end-5);
    check_ = true;
end
try
    im = imread(fullfile(imdb_info.im_path_root{1}, [process_prefix im_name_ext]));
    im_name_complete = fullfile(imdb_info.im_path_root{1}, [process_prefix im_name_ext]);
catch
    im = imread(fullfile(imdb_info.im_path_root{2}, [process_prefix im_name_ext]));
    im_name_complete = fullfile(imdb_info.im_path_root{2}, [process_prefix im_name_ext]);
end
if check_
    im = fliplr(im);
    im_name_complete = [im_name_complete(1:end-5) '_flip.JPEG'];
end
if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
target_size = conf.scales(random_scale_inds);
im_resize = prep_im_for_blob(im, target_size, conf);

processed_ims{1} = im_resize;
im_blob = im_list_to_blob(processed_ims);
end

function anchors = template_map_to_real_anchor(pattern, im_size)

anchors = cell(size(pattern));

for i = 1:length(anchors)
    anchor_num = size(pattern{i}, 1);
    temp_anchor = pattern{i} .* ...
        (repmat([im_size(2) im_size(1) im_size(2) im_size(1)], [anchor_num 1]));
    anchors{i} = ...
        [ -(temp_anchor(:, 3) - temp_anchor(:, 1))/2, ...
        -(temp_anchor(:, 4) - temp_anchor(:, 2))/2, ...
        (temp_anchor(:, 3) - temp_anchor(:, 1))/2, ...
        (temp_anchor(:, 4) - temp_anchor(:, 2))/2 ];
    anchors{i} = double(floor(anchors{i}));
end
end