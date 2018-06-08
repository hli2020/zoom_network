function [multiple_nms_boxes, raw_boxes, raw_boxes_roi] = ...
    proposal_im_detect(conf, caffe_solver, sub_im_info)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%  ULTIMATE VERSION: multi-level, multi-scale, multi-GPU (changed to single GPU mode)
%
%  raw_boxes:            cell [im_subset_num x test_scale_num]
%                                   each entry [(anchor_num_l1 + anchor_num_l2 + ...) x 5]
%
%  multiple_nms_boxes:   cell [im_subset_num x test_scale_num x nms_overlap_thres_range]
%                                   each entry [topN x 5]

test_scale_num          = length(conf.test_scale_range);
level_num               = length(conf.anchors);
gpu_num                 = length(conf.gpu_id);
nms_ov_thres            = conf.test.nms_thres;
debug                   = conf.debug;
% output
im_num                  = length(sub_im_info.imdb_merge_path);
multiple_nms_boxes      = cell(im_num, test_scale_num, length(nms_ov_thres));
raw_boxes               = cell(im_num, test_scale_num);
raw_boxes_roi           = cell(level_num, test_scale_num);

% filter out some low_score box
filer_out_low_score_box = true;
low_thres = 0.01;
if conf.roi_followup, low_thres = 0; end
keepTopBeforeNMS = conf.test.keepTopBeforeNMS;
nms_keep_after_level = conf.test.nms_keep_at_level;

%% multi-test-scale
for scale_iter = 1 : test_scale_num
    
    net_inputs = cell(gpu_num, 1);
    ori_im_size = zeros(gpu_num, 2);
    scaled_im_size = zeros(gpu_num, 2);
    curr_scale = conf.test_scale_range(scale_iter);
    
    if debug, tic; end
    % 1. prepare the input on a given scale
    for i = 1:gpu_num
        try
            im_path = fullfile(sub_im_info.im_path_root{1}, [sub_im_info.imdb_merge_path{i} '.jpg']);
            curr_im = single(imread(im_path));
            im_resize = prep_im_for_blob(curr_im, curr_scale, conf);
            
            ori_im_size(i, :) = [size(curr_im, 1) size(curr_im, 2)];
            scaled_im_size(i, :) = [size(im_resize, 1) size(im_resize, 2)];
            % permute data into caffe c++ memory, thus [num, channels, height, width]
            im_blob = im_resize(:, :, [3, 2, 1], :); % from rgb to brg
            im_blob = permute(im_blob, [2, 1, 3, 4]);
            im_blob = single(im_blob);
            
            if conf.roi_followup
                [rois_blob, conf.temp_raw_rescale_rois, origin_boxes] = ...
                    prepare_test_rois_blob(...
                    conf, conf.temp_im_subset_name{i}, ...
                    ori_im_size(i, 1:2), scaled_im_size(i, :));
            else
                net_inputs{i} = {im_blob};
            end
        catch
            % some cases length(im) < gpu_num,
            % just use the result of last iteration
            net_inputs{i} = {im_blob};
        end
    end
    if ~conf.check_certain_scale
        fprintf('\tcurr_scale %d\n', curr_scale);
    else
        % TODO
    end
    
    % 2. caffe forward pass
    if conf.roi_followup
        assert(gpu_num == 1);
        output_blobs = run_caffe_forward_many_times(caffe_solver, ...
            conf, im_blob, rois_blob, level_num);
    else
        caffe_solver.reshape_as_input(net_inputs);
        caffe_solver.forward(net_inputs);
        output_blobs = caffe_solver.get_output();
    end
    if debug
        t = toc; fprintf('caffe forwad test, scale=%d:\t\t%.4f\n', curr_scale, t); t1 = tic;
        process_time = 0; nms_time = 0;
    end
    
    %     temp{1} = 'inception_3b/output';
    %     temp{2} = 'inception_3c_mirror_upsample';
    %     temp{3} = 'conv_proposal1';
    %     temp{4} = 'inception_4d/output';
    %     temp{5} = 'inception_4e_mirror_upsample';
    %     temp{6} = 'conv_proposal2';
    %     temp{7} = 'inception_d2_para/output';
    %     temp{8} = 'inception_5b/output';
    %     temp{9} = 'conv_proposal3';
    %     featMap = cell(length(temp), 1);
    %     for feat_iter = 1:length(featMap)
    %         featMap{feat_iter} = caffe_solver.nets{1}.blobs(temp{feat_iter}).get_data;
    %     end
    %     save(sprintf('./d15a_featMap_scale_%04d.mat', curr_scale), 'featMap');
    
    % 3. process the output blob
    for i = 1:im_num % note: not gpu_num!
        
        curGPU_info.output_blob = output_blobs{i};
        curGPU_info.ori_im_size = ori_im_size(i, :);
        curGPU_info.scaled_im_size = scaled_im_size(i, :);
        
        if debug, process_per = tic; end
        if conf.roi_followup, raw_boxes_roi{1, scale_iter} = origin_boxes; end
        
        for kk = 1:level_num
            conf.temp_curr_level = kk;
            [curr_m_output, ~] = process_test_output(conf, curGPU_info);
            
            if filer_out_low_score_box
                curr_m_output = filer_out_hyli(curr_m_output, low_thres, keepTopBeforeNMS);
            end
            % accumulate raw_results in each level
            raw_boxes{i, scale_iter} = [raw_boxes{i, scale_iter}; curr_m_output];            
            if conf.roi_followup, raw_boxes_roi{kk+1, scale_iter} = curr_m_output; end
        end
        if debug
            process_time = process_time + toc(process_per); nms_per = tic;
        end
        
        % first NMS (across level)
        % multiple_nms_boxes(i, scale_iter, :) = cellfun(@(x) boxes_filter(raw_boxes{i, scale_iter}, ...
        %    x, nms_keep_after_level), ...
        %    num2cell(nms_ov_thres), 'uniformoutput', false);
        for thres = 1:length(nms_ov_thres)
            
            if ~conf.test.box_vote
                multiple_nms_boxes{i, scale_iter, thres} = ...
                    boxes_filter(raw_boxes{i, scale_iter}, nms_ov_thres(thres), nms_keep_after_level);
            else
                [~, nms_choose_ind] = ...
                    boxes_filter(raw_boxes{i, scale_iter}, nms_ov_thres(thres), nms_keep_after_level);
                multiple_nms_boxes{i, scale_iter, thres} = box_vote(...
                    raw_boxes{i, scale_iter}, nms_choose_ind, conf.test.box_vote_ov_thres);
            end
            
        end
        
        if debug, nms_time = nms_time + toc(nms_per); end
    end
    
    if debug
        fprintf('process alone, scale=%d:\t\t%.4f\n', curr_scale, process_time);
        fprintf('nms alone, scale=%d:\t\t\t%.4f\n', curr_scale, nms_time);
        t = toc(t1); fprintf('multiple nms, scale=%d:\t\t%.4f\n', curr_scale, t);
    end
    
end
end

function output_blobs = run_caffe_forward_many_times(caffe_solver, conf, im_blob, rois_blob, level_num)

% for roi-followup fast-rcnn style, allowing one gpu during test.
total_rois = size(rois_blob, 4);
chunk_length = conf.rois_per_iter;
output_blobs{1}(1).blob_name = '';
output_blobs{1}(1).data = [];

for i = 1:ceil(total_rois / chunk_length)
    
    sub_ind_start = 1 + (i-1) * chunk_length;
    sub_ind_end = min(total_rois, i * chunk_length);
    sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
    
    if level_num == 3
        sub_net_inputs{1} = {im_blob, sub_rois_blob, sub_rois_blob, sub_rois_blob};
    elseif level_num == 1
        sub_net_inputs{1} = {im_blob, sub_rois_blob};
    end
    caffe_solver.reshape_as_input(sub_net_inputs);
    caffe_solver.forward(sub_net_inputs);
    curr_output_blobs = caffe_solver.get_output();
    
    for m = 1:length(curr_output_blobs{1})
        output_blobs{1}(m).blob_name = curr_output_blobs{1}(m).blob_name;
        output_blobs{1}(m).data = cat(2, output_blobs{1}(m).data, curr_output_blobs{1}(m).data);
    end
end
end

function [rois_blob, raw_rescale_rois, origin_boxes] = ...
    prepare_test_rois_blob(conf, im_name, im_size, im_size_resize)

ld = find_rpn_box_and_load(conf.rpn_box_path('val2'), [im_name '.mat']);
origin_boxes = ld.aboxes;
rpn_boxes = origin_boxes(:, 1:4);
rois = double(scale_rois(rpn_boxes, im_size, im_size_resize));

rois_blob = [ones(size(rois, 1), 1), floor(rois)];
rois_blob = rois_blob - 1; % to c's index (start from 0)
rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
raw_rescale_rois = rois;
end

function ld = find_rpn_box_and_load(root_folder, file_name)
% file_name
%   'ILSVRC2012_val_00000001'
%   '(without ILSVRC2014_train_0006/) ILSVRC2014_train_00060655'
%   '(without ILSVRC2013_train/n07753275/) n07753275_3501'
ld = [];
% must find the original boxes (D15a, for example)
which_set = strrep(fileparts(fileparts(fileparts(root_folder))), ...
    fileparts(fileparts(fileparts(fileparts(root_folder)))), '');
which_set = which_set(2:end);
assert(strcmp(which_set, 'ilsvrc14_val2'));

if file_name(1) == 'n'
    root_folder = strrep(root_folder, which_set, 'train_liuyu_addition');
    wnid = file_name(1:9);
    new_file_name = sprintf('ILSVRC2013_train/%s/%s', wnid, file_name);
    ld = load(fullfile(root_folder, new_file_name));
end

if strcmp(file_name(7:10), '2014')
    root_folder = strrep(root_folder, which_set, 'ilsvrc14_train14');
    new_file_name = [file_name(1:end-8) '/' file_name];
    ld = load(fullfile(root_folder, new_file_name));
end

if strcmp(file_name(7:14), '2012_val') || strcmp(file_name(7:14), '2013_val')
    try
        % val2
        ld = load(fullfile(root_folder, file_name));
    catch
    end
    try
        % val1
        new_root_folder = strrep(root_folder, which_set, 'ilsvrc14_val1');
        ld = load(fullfile(new_root_folder, file_name));
    catch
    end
    
    try
        % val_liuyu_addition
        new_root_folder = strrep(root_folder, which_set, 'val_liuyu_addition');
        ld = load(fullfile(new_root_folder, file_name));
    catch
    end
end

if isempty(ld)
    error('you are screwed up: rpn boxes are not found!');
end
end

function output = filer_out_hyli(input, its_thres, keepTopBeforeNMS)
output = input(input(:, end) >= its_thres, :);
try
    output = output(1 : min(keepTopBeforeNMS, size(output, 1)), :);
catch
end
end


