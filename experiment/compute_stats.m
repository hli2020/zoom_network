% Object proposals using deep models
%   Author:             Hongyang Li
%   Affiliation:        Chinese Univ. of Hong Kong
%   Date:               Oct 3rd, 2016
%   Email:              yangli@ee.cuhk.edu.hk

% compute stats of training samples
caffe.reset_all();
close all;
clear;
mkdir_if_missing('./output/stats');
draw_boxes                              = false; % for train only

% will lower down bg_thres_hi if turned on, see 'init_rpn.m'
conf.add_gray_cls                       = true; % maybe unused during test

% result_prefix                           = 'new2';
% result_prefix                           = 'new_anchorTemplate_opt1';
% result_prefix                           = 'dense_wise';

% % if turned on, anchors are generated on-the-fly in each image
% conf.use_new_anchor                     = false;
% ld = load('./anchors_opt1.mat');
% conf.anchor_pattern = ld.anchors; clear ld;

%% collect recall and gt_facts
ratio_space{1} = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
ratio_space{2} = [0.5, 1, 2, 5, 10, 20];
ratio_space{3} = [0.05, 0.1, 0.2, 0.5, 1, 2];
ratio_space{4} = [0.05, 0.2, 0.5, 1, 2, 5, 20];
ratio_space{5} = [0.1, 0.2, 0.5, 1, 2, 5, 10];
base_size_space = 8:3:30;

range_down = 31;
range_up = 40;

cnt = 0;
for kk = 1:length(ratio_space)
    for j = 1:length(base_size_space)
        
        cnt = cnt + 1;       
        if cnt <= range_up && cnt >= range_down
            result_prefix = sprintf('dense_greedy_search_%d', cnt);
            % train
            conf_in                                 = [];
            conf_in.rpn_max_size                    = 1200; % larger size will be out of memory
            % conf_in.ratios                          = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
            conf_in.ratios = ratio_space{kk};
            % conf_in.base_size                       = 8; %16*sqrt(2);
            conf_in.base_size = base_size_space(j);
            conf_in.anchor_scale                    = 2.^(0:8); % 6 x 3 = 18 anchors
            adjust_sample_ratio                     = true;
            
            %% configure: dataset, model, params
            conf.model_id                           = 'XYZ_compute_stats_new2';
            conf.multi_depth                        = true;
            conf.check_certain_scale                = true;
            
            conf.train.arch                         = 'bn';
            conf.test.nms_mode                      = 'normal_nms';
            conf.test.multi_scale                   = true;
            conf.test.res_folder_suffix             = 'multiLoss_multiScale';
            conf.test_scale_range                   = (1300:-100:300);
            conf.test_multiscale_max_size           = 1500;
            
            if conf.multi_depth
                conf.train.solver_file              = 'set_4_hourglass/solver_train';
                conf.test.solver_file               = 'set_4_hourglass/solver_deploy';
                conf_in.rpn_feat_stride             = [8, 16, 32];
                conf.test.multi_level_useInd        = [1 1 1];
            end
            conf_in.compute_stats = true;
            [model, conf, dataset] = init_rpn(conf, conf_in);
            
            %%
            % collect imdb, roidb info
            imdbs = dataset.imdb_train;
            roidbs = dataset.roidb_train;
            imdb_info = collect_db_info(imdbs, roidbs);
            conf.rpn_param.draw_boxes = draw_boxes;
            conf.rpn_param.adjust_sample_ratio  = adjust_sample_ratio;
            conf.rpn_param.model_id = conf.model_id;
            conf.rpn_param.compute_stats = conf_in.compute_stats;
            try
                conf.rpn_param.anchor_pattern = conf.anchor_pattern;
                conf.rpn_param.use_new_anchor = conf.use_new_anchor ;
            catch
                conf.rpn_param.use_new_anchor = false;
            end
            recall.all = zeros(200, 3);
            recall.level_1 = zeros(200, 3);
            recall.level_2 = zeros(200, 3);
            recall.level_3 = zeros(200, 3);
            gt_resize_log = cell(200, 2);
            
            for i = 1:2:length(imdb_info.roidb_merge)
                
                close all;
                input.roidb_merge = imdb_info.roidb_merge(i);
                input.imdb_merge_size = imdb_info.imdb_merge_size(i, :);
                input.imdb_merge_path{1} = imdb_info.imdb_merge_path{i};
                input.im_path_root = imdb_info.im_path_root;
                conf.rpn_param.temp_which_ind = (i+1)/2;
                [~, gt_info, gt_stats] = ...
                    proposal_generate_minibatch_on_the_fly(conf.rpn_param, input);
                [recall, gt_resize_log] = update_recall_stats(...
                    recall, gt_resize_log, gt_info, gt_stats);
                
                if i == 1 || (i+1)/2 == (length(imdb_info.roidb_merge)/2) ...
                        || mod(conf.rpn_param.temp_which_ind, 1000)== 0
                    fprintf('i/total:: %d/%d\n', ...
                        conf.rpn_param.temp_which_ind, length(imdb_info.roidb_merge)/2);
                end
            end
            
            total_inst = recall.all(:, 1);
            correct_inst = recall.all(:, 2);
            recall.all(:, 3) = correct_inst ./ total_inst;
            
            final_av_recall = sum(correct_inst(:)) / sum(total_inst(:));
            recall.level_1(:, 3) = recall.level_1(:, 2) ./ recall.level_1(:, 1);
            recall.level_2(:, 3) = recall.level_2(:, 2) ./ recall.level_2(:, 1);
            recall.level_3(:, 3) = recall.level_3(:, 2) ./ recall.level_3(:, 1);
            
            save(sprintf('./output/stats/%s_train_recall.mat', result_prefix), ...
                'recall', 'final_av_recall', 'conf');
            %     save(sprintf('./output/stats/%s_train_resize_gt.mat', result_prefix), 'gt_resize_log');
        end
    end
end


%% compute gt stats and find their pattern by kmeans
if 0
    all_sample_box_normalized = [];
    all_sample_box_scale = [];
    stats_per_cls(200).norm_box = [];
    % input
    ld = load(sprintf('./output/stats/new2_train_resize_gt.mat'));
    gt_resize_log = ld.gt_resize_log; clear ld;
    for i = 1:200
        fprintf('cls %d\n', i);
        im_resize = gt_resize_log{i, 2};
        temp = [im_resize(:, 2) im_resize(:, 1) ...
            im_resize(:, 2) im_resize(:, 1)];
        
        stats_per_cls(i).norm_box = gt_resize_log{i, 1} ./ temp;
        stats_per_cls(i).scale = ...
            (stats_per_cls(i).norm_box(:, 3) - stats_per_cls(i).norm_box(:, 1)) .* ...
            (stats_per_cls(i).norm_box(:, 4) - stats_per_cls(i).norm_box(:, 2));
        stats_per_cls(i).aspect_ratio = ... % W/H
            (stats_per_cls(i).norm_box(:, 3) - stats_per_cls(i).norm_box(:, 1)) ./ ...
            (stats_per_cls(i).norm_box(:, 4) - stats_per_cls(i).norm_box(:, 2));
        
        all_sample_box_normalized = cat(1, ...
            all_sample_box_normalized, ...
            stats_per_cls(i).norm_box);
        
        all_sample_box_scale = cat(1, ...
            all_sample_box_scale, ...
            stats_per_cls(i).scale);
    end
    for i = 1:200
        stats_per_cls(i).scale_mean = mean(stats_per_cls(i).scale);
        stats_per_cls(i).scale_std = std(stats_per_cls(i).scale);
        stats_per_cls(i).aspect_ratio(isinf(stats_per_cls(i).aspect_ratio)) = 0;
        stats_per_cls(i).aspect_ratio_mean = mean(stats_per_cls(i).aspect_ratio);
        stats_per_cls(i).aspect_ratio_std = std(stats_per_cls(i).aspect_ratio);
    end
    %     save(sprintf('./output/stats/%s_stats.mat', result_prefix), ...
    %         'all_sample_box_normalized', 'all_sample_box_scale', 'stats_per_cls');
    
    % option1: kmeans using all traning samples
    K = 30;
    opts = statset('Display', 'iter', 'maxiter', 500);
    [~, center1] = kmeans(all_sample_box_normalized, K, 'options', opts);
    % save(sprintf('./output/stats/%s_kmeans.mat', result_prefix), ...
    %     'center');
    temp_scale = (center1(:, 1) - center1(:, 3)).*(center1(:, 2) - center1(:, 4));
    [watch_sort_scale1, ind1] = sort(temp_scale);
    center1 = center1(ind1, :);
    anchors = cell(3, 1);
    for i = 1:3
        window = 1 + (i-1)*10 : 10 + (i-1)*10;
        anchors{i} = center1(window, :);
    end
    save('anchors_opt1.mat', 'anchors');
    
    % option2: first divide the 'all_sample_box_scale' into 3 levels and then
    % cluster within the grouped data
    [~, ind] = sort(all_sample_box_scale);
    sort_all_sample_box_normalized = all_sample_box_normalized(ind, :);
    interval = floor(length(all_sample_box_scale)/3);
    K = 10;
    opts = statset('Display', 'iter', 'maxiter', 500);
    center2 = [];
    for i = 1:3
        window = 1+(i-1)*interval : min(interval+(i-1)*interval, length(all_sample_box_scale));
        [~, curr_center] = kmeans(sort_all_sample_box_normalized(window, :), K, 'options', opts);
        center2 = [center2; curr_center];
    end
    temp_scale = (center2(:, 1) - center2(:, 3)).*(center2(:, 2) - center2(:, 4));
    [watch_sort_scale2, ind2] = sort(temp_scale);
    
    anchors = cell(3, 1);
    for i = 1:3
        window = 1 + (i-1)*10 : 10 + (i-1)*10;
        anchors{i} = center2(window, :);
    end
    save('anchors_opt2.mat', 'anchors');
end
