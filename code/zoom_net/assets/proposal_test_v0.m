function output = proposal_test_v0(conf, imdb, varargin)
output.raw_aboxes = [];

ip = inputParser;
ip.addRequired('conf',                              @isstruct);
ip.addRequired('imdb',                              @isstruct);
ip.addParameter('solver_def_file',                  @isstr);
ip.addParameter('trained_model',                    @isstr);
ip.addParameter('output_raw_box',       false,      @islogical);
ip.addParameter('rpn_box_nms_list',                 @iscell);
ip.addParameter('nms',                              @isstruct);
ip.parse(conf, imdb, varargin{:});
opts = ip.Results;

nms_overlap_thres           = opts.nms.nms_overlap_thres;
% e.g., output/rpn/D03_s31/ilsvrc14_val2/iter_1000_suffix/nms_0.65/split
rpn_box_after_nms_list      = opts.rpn_box_nms_list;
output_rawBox_switch        = opts.output_raw_box;
root_dir                    = fileparts(fileparts(rpn_box_after_nms_list{1}));
iter                        = strrep(root_dir, [fileparts(root_dir) '/'], '');
debug                       = conf.debug;

conf.folder_suffix{1}       = 'stack2';
conf.folder_suffix{2}       = 'stack1';
conf.folder_suffix{3}       = 'together';
%% init net
caffe.reset_all();
caffe.set_mode_gpu();
if ~conf.extract_prop
    % init caffe log
    mkdir_if_missing([root_dir '/caffe_log']);
    caffe_log_file_base = fullfile(root_dir, 'caffe_log/test_');
    caffe.init_log(caffe_log_file_base);
end
% init solver
caffe_solver = caffe.Solver(opts.solver_def_file, conf.gpu_id);
caffe_solver.use_caffemodel(opts.trained_model);
caffe_solver.set_phase('test');
% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

disp('opts:'); disp(opts); disp('conf:'); disp(conf);
%% setup
box_result          = [];   % unique to extract-prop case
list                = [];   % unique to extract-prop case
im_path_book        = [];   % unique to non-extract-prop case

if conf.use_coco
    if ~isfield(imdb, 'api')
        % normal test
        full_im_list    = extractfield(imdb.coco.data.images, 'file_name')';
        full_im_list    = cellfun(@(x) x(1:end-4), full_im_list, 'uniformoutput', false);
    else
        % extract train boxes
        full_im_list = [];
        for ii = 1:length(imdb.api)
            curr_full_im_list = extractfield(imdb.api{ii}.data.images, 'file_name')';
            curr_full_im_list = cellfun(@(x) x(1:end-4), curr_full_im_list, 'uniformoutput', false);
            full_im_list = [full_im_list; curr_full_im_list];
        end
    end
    full_num_images = length(full_im_list);
else
    full_num_images = length(imdb.image_ids);
    full_im_list    = imdb.image_ids;
end
test_scale_num      = length(conf.test_scales);
gpu_num             = length(conf.gpu_id);

if output_rawBox_switch
    raw_aboxes = cell(full_num_images, test_scale_num);
end

if conf.roi_followup && conf.draw_boxes
    im_path = './data/datasets/ilsvrc14_det/ILSVRC2013_DET_val';
    gt_path = './data/datasets/ilsvrc14_det/ILSVRC2013_DET_bbox_val';
end
% note: the following two cases overlap with each other
if ~conf.extract_prop
    for i = 1:length(nms_overlap_thres)
        mkdir_if_missing(rpn_box_after_nms_list{i});
        if conf.test_inter_super
            for inter_super = 1:3
                mkdir_if_missing([rpn_box_after_nms_list{i} '/' conf.folder_suffix{inter_super}]);
            end
        end
    end
    if (gpu_num ~= 1), warning(upper('we suggest gpu number should be 1.')); end
    
    if conf.chunk_mode
        ck_interval = ceil(full_num_images/conf.test_total_chunk);
        start_ind = 1 + (conf.test_curr_chunk-1)*ck_interval;
        end_ind = min(ck_interval + (conf.test_curr_chunk-1)*ck_interval, full_num_images);
        num_images = end_ind - start_ind + 1;
        im_path_book = full_im_list(start_ind:end_ind);  % part of the whole set
    else
        num_images = full_num_images;
        im_path_book = full_im_list;  % the whole set
    end
else
    save_prefix = ['./output/extract_prop/' conf.extract_prop_param.folder_name];
    mkdir_if_missing(save_prefix);
    % also redefine the number of images
    try
        full_num_images = length(conf.extract_prop_param.train_list);
        
        % the train case, we MUST use single GPU and split into chunks
        if (gpu_num ~= 1), error('gpu number must be 1.'); end
        full_list = conf.extract_prop_param.train_list;
        ck_int = ceil(full_num_images/conf.extract_prop_param.total_chunk);
        start_ind = 1 + (conf.extract_prop_param.curr_chunk-1)*ck_int;
        end_ind = min(ck_int + (conf.extract_prop_param.curr_chunk-1)*ck_int, full_num_images);
        
        if conf.extract_prop_param.curr_chunk == 25, start_ind = 65674; end
        if conf.extract_prop_param.curr_chunk == 32, end_ind = 87564; end
        % redefined 'list' and 'num_images'
        list = full_list(start_ind:end_ind);
        num_images = end_ind - start_ind + 1;
        save_name = [save_prefix sprintf('/train_ck%d_absInd_%d_%d.mat', ...
            conf.extract_prop_param.curr_chunk, start_ind, end_ind)];
        fprintf('extracting train proposals (gpu=%d), chunk #%d, from %d to %d, (%d) images...\n', ...
            conf.gpu_id, conf.extract_prop_param.curr_chunk, start_ind, end_ind, num_images);
    catch
        num_images = length(conf.extract_prop_param.val_list);
        list = conf.extract_prop_param.val_list;
        save_name = [save_prefix '/val.mat'];
        fprintf('extracting val proposals, (%d) images...\n', num_images);
    end
    box_result(num_images).name = '';
    box_result(num_images).box = [];
end

cnt = 0; % count the actual number of images
show_num = 200;
iter_length = ceil(num_images/gpu_num);

%% testing
% when gpu_num = 1, iter_length = num_images
% for i = 6422%1:iter_length
for i = 1:iter_length
    
    start_im = 1 + (i-1)*gpu_num;
    end_im = min(gpu_num + (i-1)*gpu_num, num_images);
    cnt = cnt + end_im - start_im + 1;
    
    if conf.chunk_mode
        % ck#1(1-22439), test folder: iter_170000_dense_test, 	1 curr / 22439 total,	GPU=0, (5) nms thres, (24) test scale ...
        if i == 1 || cnt == num_images || mod(cnt, show_num)==0
            fprintf('ck#%d(%d-%d), %s: %s, \t%d curr / %d total,\tGPU=%d, (%d) nms thres, (%d) test scale ... \n', ...
                conf.test_curr_chunk, start_ind, end_ind, imdb.name, iter, cnt, num_images, conf.gpu_id, ...
                length(nms_overlap_thres), test_scale_num);
        end
    else
        % 'test     iter: iter_5000_dense, 78/9240,  (1) GPUs, (1) nms thres, (24) test scale ...'
        if i == 1 || cnt == num_images || mod(cnt, show_num)==0
            fprintf('test\titer: %s, \t%d/%d\t\t(%d) GPUs, (%d) nms thres, (%d) test scale ... \n', ...
                iter, cnt, num_images, gpu_num, ...
                length(nms_overlap_thres), test_scale_num);
        end
    end
    
    sub_im_list = (start_im:end_im);    % absolute index
    if ~conf.extract_prop
        [RUN_THIS_ITER, proposals] = detect_if_split_file_exist(sub_im_list, ...
            nms_overlap_thres, rpn_box_after_nms_list, conf, im_path_book);
    else
        RUN_THIS_ITER = true;
    end
    
    if RUN_THIS_ITER
        
        timing = tic;
        [im_subset, conf.temp_im_subset_name] = ...
            prepare_im_set(conf, imdb, sub_im_list, ...
            im_path_book, ... % used in non_extract_prop case
            box_result, list, start_im); % used in extract_prop case
        
        fprintf('iter %d / %d\n', i, iter_length);
        if debug, hehe = tic; end
        
        % multiple_nms_boxes: cell type
        %  [ (maybe: inter_super x) im_subset_num x test_scale_num x nms_overlap_thres_range ]
        %  each entry [topN x 5]
        %  "conf.raw_boxes_roi" is the to-be-processed variable
        if ~conf.test_inter_super
            [multiple_nms_boxes, raw_aboxes(sub_im_list, :), conf.raw_boxes_roi] = ...
                proposal_im_detect(conf, caffe_solver, im_subset, opts.nms);
        else
            multiple_nms_boxes = proposal_im_detect(conf, caffe_solver, im_subset, opts.nms);
        end
        
        if debug
            t = toc(hehe); cprintf('blue', 'proposal_im_detect:: %.4f\n', t);
            hehe = tic;
        end
        
        if conf.test_roi_direct_to_final
            % roi-followup test procedure
            split_and_save_roi_direct(conf, rpn_box_after_nms_list, [im_path_book{sub_im_list(1)} '.mat']);
        else
            % normal case: will do a second nms
            % 'result' is for (conf.extract_prop) and (conf.roi_followup &&
            % conf.draw_boxes)
            result = split_and_save(sub_im_list, nms_overlap_thres, ...
                multiple_nms_boxes, rpn_box_after_nms_list, opts.nms, conf, im_path_book);
        end
        
        if conf.extract_prop
            % for now, we only allow 1 nms thres for 'extract_prop' case
            % NOTE: we haven't checked if the following is right under the
            % 'intermediate supversion' case
            assert(size(result, 2)==1);
            for kk = 1:length(sub_im_list)
                box_result(sub_im_list(kk)).box = ...
                    result{kk}(1: min(conf.extract_prop_param.top_k, size(result{kk}, 1)), 1:4);
            end
        end
        if debug, t = toc(hehe); cprintf('blue', 'split_and_save:: %.4f\n', t); end
        
        timing_end = toc(timing);
        if ~debug, cprintf('blue', 'remaining::\t\t%.2f hrs\n\n', ...
                (iter_length-i)*(timing_end/3600)); end
        
        if conf.roi_followup && conf.draw_boxes
            % prepare variable 'proposals'
            proposals.raw_boxes_roi = conf.raw_boxes_roi;
            proposals.aboxes = result';     % cell column vec
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

if output_rawBox_switch, output.raw_aboxes = raw_aboxes; end
if conf.extract_prop, save(save_name, 'box_result'); end

diary off;
caffe.reset_all();
rng(prev_rng);
end

function [RUN_THIS_ITER, proposals] = detect_if_split_file_exist(sub_im_list, ...
    nms_overlap_thres, rpn_box_after_nms_list, conf, im_path_book)

RUN_THIS_ITER = false;
% for 'conf.roi_followup && conf.draw_boxes' use
proposals.aboxes = cell(length(nms_overlap_thres), 1);
proposals.raw_boxes_roi = [];

if ~conf.test_inter_super
    
    % normal case
    for kk = 1:length(sub_im_list)
        for xx = 1:length(nms_overlap_thres)
            
            RUN_THIS_ITER = find_this_file(rpn_box_after_nms_list{xx}, ...
                [im_path_book{sub_im_list(kk)} '.mat']);
            if RUN_THIS_ITER, break; end
            %             if ~exist(fullfile(rpn_box_after_nms_list{xx}, ...
            %                     [im_path_book{sub_im_list(kk)} '.mat']), 'file')
            %                 RUN_THIS_ITER = true;
            %                 break;
            %             end
            try
                ld = load(fullfile(rpn_box_after_nms_list{xx}, ...
                    [im_path_book{sub_im_list(kk)} '.mat']));
                proposals.aboxes{xx} = ld.aboxes;
                proposals.raw_boxes_roi = ld.raw_boxes_roi;
            catch
            end
        end
        if RUN_THIS_ITER, break; end
    end
else
    % intermediate supervision case
    for inter_super = 1:3
        for kk = 1:length(sub_im_list)
            for xx = 1:length(nms_overlap_thres)
                if ~exist(fullfile(rpn_box_after_nms_list{xx}, conf.folder_suffix{inter_super}, ...
                        [im_path_book{sub_im_list(kk)} '.mat']), 'file')
                    RUN_THIS_ITER = true;
                    break;
                end
            end
            if RUN_THIS_ITER, break; end
        end
        if RUN_THIS_ITER, break; end
    end
end
end

function flag = find_this_file(root_folder, file_name)

flag = true;
% file_name
%   '(val/)ILSVRC2012_val_00000001'
%   '(train/)ILSVRC2014_train_0006/ILSVRC2014_train_00060655'
%   '(train/)ILSVRC2013_train/n07753275/n07753275_3501'

% to search scope order:
%   itself (train/val_liuyu_addition)
search_set{1} = 'ilsvrc14_val1';
search_set{2} = 'ilsvrc14_val2';
search_set{3} = 'ilsvrc14_train14';

which_set = strrep(fileparts(fileparts(fileparts(root_folder))), ...
    fileparts(fileparts(fileparts(fileparts(root_folder)))), '');
which_set = which_set(2:end);

if file_name(1) == 't', file_name = file_name(7:end); end
if file_name(1) == 'v', file_name = file_name(5:end); end

if exist(fullfile(root_folder, file_name), 'file')
    flag = false; return;
else
    for i = 1:length(search_set)
        if exist(fullfile(strrep(root_folder, which_set, search_set{i}), ...
                file_name), 'file')
            flag = false; return;
        end
    end
end
end

function multiple_nms_boxes_new = split_and_save(sub_im_list, nms_overlap_thres, ...
    multiple_nms_boxes, rpn_box_after_nms_list, nms, conf, im_path_book)
% for conf.extract_prop case use
multiple_nms_boxes_new = [];

if ~conf.test_inter_super
    
    multiple_nms_boxes_new = process_(multiple_nms_boxes, conf, nms, nms_overlap_thres);
    if ~conf.extract_prop
        % save the split files
        for kk = 1:length(sub_im_list)
            for xx = 1:length(nms_overlap_thres)
                aboxes = multiple_nms_boxes_new{kk, xx};
                to_save_file = adjust_save_file_name(rpn_box_after_nms_list{xx}, ...
                    [im_path_book{sub_im_list(kk)} '.mat']);
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
    
else
    for inter_super = 1:3
        curr_output = process_(squeeze(multiple_nms_boxes(inter_super, :, :, :)), ...
            conf, nms, nms_overlap_thres);
        % save the split files
        for kk = 1:length(sub_im_list)
            for xx = 1:length(nms_overlap_thres)
                aboxes = curr_output{kk, xx};
                to_save_file = fullfile(rpn_box_after_nms_list{xx}, conf.folder_suffix{inter_super}, ....
                    [im_path_book{sub_im_list(kk)} '.mat']);
                save(to_save_file, 'aboxes', '-v7.3');
            end
        end
    end
end
end

function split_and_save_roi_direct(conf, rpn_box_after_nms_list, file_name)

to_process_box = conf.raw_boxes_roi;
out = process_regression_result([], to_process_box, 'naive_2_nms7');
aboxes = out{1};
to_save_file = adjust_save_file_name(rpn_box_after_nms_list{1}, file_name);
mkdir_if_missing(fileparts(to_save_file));
save(to_save_file, 'aboxes', '-v7.3');
end

function to_save_file_name = adjust_save_file_name(folder, file_name)
% file_name
%   '(val/)ILSVRC2012_val_00000001' (hope this never or rarely happens in *_liuyu set)
%   '(train/)ILSVRC2014_train_0006/ILSVRC2014_train_00060655' (will save to the ilsvrc14_train14 folder)
%   '(train/)ILSVRC2013_train/n07753275/n07753275_3501' (itself)
% to save scope list:
%   itself (train/val_liuyu_addition)
%   ilsvrc14_train14

if file_name(1) == 't', file_name = file_name(7:end); end
if file_name(1) == 'v', file_name = file_name(5:end); end

if strcmp(file_name(7:10), '2014')
    
    replace_set = 'ilsvrc14_train14';
    which_set = strrep(fileparts(fileparts(fileparts(folder))), ...
        fileparts(fileparts(fileparts(fileparts(folder)))), '');
    which_set = which_set(2:end);
    to_save_file_name = fullfile(strrep(folder, which_set, replace_set), file_name);
else
    to_save_file_name = fullfile(folder, file_name);
end

end

function multiple_nms_boxes_new = process_(multiple_nms_boxes, conf, nms, nms_overlap_thres)

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
            if ~conf.box_vote
                % use the same thres as that in merging different levels
                multiple_nms_boxes_new{kk, xx} = ...
                    boxes_filter_inline(all_scale_result, nms.per_nms_topN, ...
                    nms_overlap_thres(xx), nms.after_nms_topN, true);
            else
                [~, nms_choose_ind] = boxes_filter_inline(all_scale_result, ...
                    nms.per_nms_topN, nms_overlap_thres(xx), nms.after_nms_topN, true);
                multiple_nms_boxes_new{kk, xx} = box_vote(...
                    all_scale_result, nms_choose_ind, conf.box_vote_ov_thres);
            end
        end
    end
end
end

function [im_subset, im_subset_name] = ...
    prepare_im_set(conf, imdb, sub_im_list, im_path_book, box_result, list, start_im)

im_subset = cell(length(sub_im_list), 1);
im_subset_name = cell(length(sub_im_list), 1);

for kk = 1:length(sub_im_list)
    
    %     if ~conf.extract_prop
    file_name = [im_path_book{sub_im_list(kk)}, '.', imdb.extension];
    if file_name(1) == 't'
        im_path = fullfile('./data/datasets/ilsvrc14_det/ILSVRC2014_DET_train', file_name(7:end));
    elseif file_name(1) == 'v'
        im_path = fullfile('./data/datasets/ilsvrc14_det/ILSVRC2013_DET_val', file_name(5:end));
    elseif strcmp(file_name(1:6), 'COCO_t')
        im_path = fullfile(imdb.image_dir{2}, file_name);
    elseif strcmp(file_name(1:6), 'COCO_v')
        im_path = fullfile(imdb.image_dir{1}, file_name);
    else
        im_path = fullfile(imdb.image_dir, file_name);
    end
    %     else
    %         % WILL BE DEPRECATED
    %         % specify im path in the 'extract_prop' case
    %         name_temp = list{start_im + kk -1}(21:end);
    %         box_result(start_im + kk -1).name = name_temp;
    %         if name_temp(1) == 't'
    %             im_path = fullfile('./data/datasets/ilsvrc14_det/ILSVRC2014_DET_train', name_temp(7:end));
    %         elseif name_temp(1) == 'v'
    %             im_path = fullfile('./data/datasets/ilsvrc14_det/ILSVRC2013_DET_val', name_temp(5:end));
    %         end
    %     end
    
    try
        im_subset{kk} = imread(im_path);
    catch lasterror
        % hah, annoying data issues
        if strcmp(lasterror.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
            warning('converting %s from CMYK to RGB', im_path);
            cmd = ['convert ' im_path ' -colorspace CMYK -colorspace RGB ' im_path];
            system(cmd);
            im_subset{kk} = imread(im_path);
        else
            error(lasterror.message);
        end
    end
    [~, im_subset_name{kk}, ~] = fileparts(im_path);
    if size(im_subset{kk}, 3) == 1
        im_subset{kk} = repmat(im_subset{kk}, [1 1 3]);
    end
end
end
