function zoom_test(varargin)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%
%   Note:
%           conf.extract_prop is INACTIVE
%           conf.test_roi_direct_to_final is INACTIVE (roi-followup test procedure)
%           conf.test_inter_super is INACTIVE (intermediate supervision case)

ip = inputParser;
ip.addParameter('conf',                     @isstruct);
ip.addParameter('solver',                   @isstr);
ip.addParameter('test_model',               @isstr);
ip.parse(varargin{:});
opts = ip.Results;

%% init
conf                                = opts.conf;
debug                               = conf.debug;
chunk_mode                          = conf.chunk_mode;

check_folder = fullfile(pwd, 'output', conf.model_id(1:end-5));
if exist(check_folder, 'dir')
    cache_dir = fullfile(pwd, 'output', conf.model_id(1:end-5), 'test');
else
    cache_dir = fullfile(pwd, 'output', conf.model_id, 'test');
end
[~, iter_name] = fileparts(opts.test_model);
conf.iter_name = strrep(iter_name, '.caffemodel', '');
mkdir_if_missing(cache_dir);
% init solver
sovler_file = fullfile(pwd, 'model', 'zoom', [opts.solver '.prototxt']);
caffe_solver = caffe.Solver(sovler_file, conf.gpu_id);
caffe_solver.use_caffemodel(opts.test_model);
caffe_solver.set_phase('test');
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

[im_path_book, num_images, res_path, start_ind, end_ind, imdb_info] = ...
    test_data_loader(conf, cache_dir);
% test parameters
test_scale = conf.test_scale_range;

%% MULTI-scale, level testing; alongside with evaluation
cnt = 0; % count the actual number of images
gpu_num = length(conf.gpu_id);
show_num = 200;
iter_length = ceil(num_images/gpu_num);

for i = 1:iter_length
    
    start_im = 1 + (i-1)*gpu_num;
    end_im = min(gpu_num + (i-1)*gpu_num, num_images);
    cnt = cnt + end_im - start_im + 1;
    
    % show progress
    if chunk_mode
        % 'ck#1 (1-22439), test model: iter_700,  1/22439, GPU=0, # nms thres: 5, # test scale: 24'
        if i == 1 || cnt == num_images || mod(cnt, show_num)==0
            fprintf('ck#%d (%d-%d), test model: %s, \t%d/%d,\tGPU=%d, # nms thres: %d, # test scale: %d\n', ...
                conf.curr_chunk, start_ind, end_ind, conf.iter_name, cnt, num_images, conf.gpu_id, ...
                length(res_path), length(test_scale));
        end
    else
        % 'test model: iter_5000_dense, 78/9240, GPU=0, # nms thres: 5, # test scale: 24'
        if i == 1 || cnt == num_images || mod(cnt, show_num)==0
            fprintf('test model: %s, \t%d/%d\t\tGPU=%d, # nms thres: %d, # test scale: %d\n', ...
                conf.iter_name, cnt, num_images, gpu_num, length(res_path), length(test_scale));
        end
    end
    
    % absolute index
    sub_im_list = (start_im:end_im);
    RUN_THIS_ITER = detect_if_split_file_exist(sub_im_list, res_path, im_path_book);
    
    if RUN_THIS_ITER
        
        t1 = tic; fprintf('iter %d / %d\n', i, iter_length);
        if debug, hehe = tic; end
        
        sub_im_info = prepare_im_info(imdb_info, sub_im_list);
        [multiple_nms_boxes, ~, conf.raw_boxes_roi] = ...
            proposal_im_detect(conf, caffe_solver, sub_im_info);
        
        if debug, t = toc(hehe); cprintf('blue', 'proposal_im_detect:: %.4f\n', t); hehe = tic; end
        
        % second NMS (across multi-scale), 'result' is for
        % (conf.extract_prop) and (conf.roi_followup && conf.draw_boxes)
        result = split_and_save(sub_im_list, im_path_book, multiple_nms_boxes, res_path, conf);
        
        if debug, t = toc(hehe); cprintf('blue', 'split_and_save:: %.4f\n', t); end
        
        timing_end = toc(t1);
        if ~debug, cprintf('blue', 'remaining::\t\t%.2f hrs\n\n', (iter_length-i)*(timing_end/3600)); end
        
        if conf.roi_followup && conf.draw_boxes
            % prepare variable 'proposals'
            proposals.raw_boxes_roi = conf.raw_boxes_roi;
            proposals.aboxes = result'; % cell column vec
        end
    end  % end RUN_THIS_ITER
    
    if conf.roi_followup && conf.draw_boxes
        % for non-coco case, too slow for computing on-the-fly on local pc
        close all;
        visual_anchor_test( im_path, im_path_book{sub_im_list}, proposals, ...
            false, [], gt_path, 0, 0, true, []);
        title('iter: %d / %d', i, iter_length);
        keyboard;
    end
end

diary off;
caffe.reset_all();
rng(prev_rng);

end

function sub_im_info = prepare_im_info(imdb_info, sub_im_list)

sub_im_info.im_path_root = imdb_info.im_path_root;
sub_im_info.imdb_merge_path = imdb_info.imdb_merge_path(sub_im_list);
sub_im_info.roidb_merge = imdb_info.roidb_merge(sub_im_list);
end

function [im_path_book, num_images, res_path, start_ind, end_ind, imdb_info] = ...
    test_data_loader(conf, cache_dir)

start_ind = [];
end_ind = [];
if strcmp(conf.dataset, 'voc')
    imdb_info = get_voc_info(conf.mode);
end
full_num_images = length(imdb_info.imdb_merge_path);
full_im_list = imdb_info.imdb_merge_path;

if conf.chunk_mode
    ck_interval = ceil(full_num_images/conf.total_chunk);
    start_ind = 1 + (conf.curr_chunk-1)*ck_interval;
    end_ind = min(ck_interval + (conf.curr_chunk-1)*ck_interval, full_num_images);
    num_images = end_ind - start_ind + 1;
    im_path_book = full_im_list(start_ind:end_ind);  % part of the whole set
else
    num_images = full_num_images;
    im_path_book = full_im_list;  % the whole set
end

% e.g., output/F01_baseline_voc/test/iter_1000/nms_0.65/
res_path = cell(length(conf.test.nms_thres), 1);
for i = 1:length(res_path)
    res_path{i} = fullfile(cache_dir, conf.iter_name, ....
        sprintf('nms_%.2f', conf.test.nms_thres(i)));
    mkdir_if_missing(res_path{i});
end
end

function RUN_THIS_ITER = detect_if_split_file_exist(sub_im_list, res_path, im_path_book)

RUN_THIS_ITER = false;
% % for 'conf.roi_followup && conf.draw_boxes' use
% proposals.aboxes = cell(length(nms_ov_thres), 1);
% proposals.raw_boxes_roi = [];

% normal case
for kk = 1:length(sub_im_list)
    for xx = 1:length(res_path)
        
        RUN_THIS_ITER = ~exist(fullfile(res_path{xx}, ...
            [im_path_book{sub_im_list(kk)} '.mat']), 'file');
        if RUN_THIS_ITER, break; end
        %         try
        %             ld = load(fullfile(res_path{xx}, ...
        %                 [im_path_book{sub_im_list(kk)} '.mat']));
        %             proposals.aboxes{xx} = ld.aboxes;
        %             proposals.raw_boxes_roi = ld.raw_boxes_roi;
        %         catch
        %         end
    end
    if RUN_THIS_ITER, break; end
end
end

function multiple_nms_boxes_new = split_and_save(sub_im_list, im_path_book, ...
    multiple_nms_boxes, res_path, conf)

% for conf.extract_prop case use
multiple_nms_boxes_new = process_(multiple_nms_boxes, conf);

% save the split files
for kk = 1:length(sub_im_list)
    for xx = 1:length(res_path)
        aboxes = multiple_nms_boxes_new{kk, xx};
        to_save_file = fullfile(res_path{xx}, [im_path_book{sub_im_list(kk)} '.mat']);
        mkdir_if_missing(fileparts(to_save_file));
        
        if conf.roi_followup
            raw_boxes_roi = conf.raw_boxes_roi;
            save(to_save_file, 'aboxes', 'raw_boxes_roi', '-v7.3');
        else
            save(to_save_file, 'aboxes', '-v7.3');
        end
    end
end

end

function multiple_nms_boxes_new = process_(multiple_nms_boxes, conf)

nms_ov_thres = conf.test.nms_thres;
nms_keep_after_level = conf.test.nms_keep_at_level;

if size(multiple_nms_boxes, 2) == 1
    % single scale test
    multiple_nms_boxes_new = squeeze(multiple_nms_boxes);
else
    % merge all scales by a further nms step
    multiple_nms_boxes_new = ...
        cell(size(multiple_nms_boxes, 1), size(multiple_nms_boxes, 3));    
    im_num = size(multiple_nms_boxes, 1);
    nms_thres_num = size(multiple_nms_boxes, 3);
    
    for kk = 1 : im_num
        for xx = 1 : nms_thres_num
            % remove empty cells
            temp = multiple_nms_boxes(kk, :, xx)';
            temp = temp(~cellfun(@isempty, temp));
            all_scale_result = cell2mat(temp);
            if ~conf.test.box_vote
                % use the same thres as that in merging different levels
                multiple_nms_boxes_new{kk, xx} = ...
                    boxes_filter(all_scale_result, nms_ov_thres(xx), nms_keep_after_level);
            else
                [~, nms_choose_ind] = boxes_filter(all_scale_result, ...
                    nms_ov_thres(xx), nms_keep_after_level);
                multiple_nms_boxes_new{kk, xx} = box_vote(...
                    all_scale_result, nms_choose_ind, conf.test.box_vote_ov_thres);
            end
        end
    end
end
end