function [catch_total, inst_total, gt_obj_fail] = visual_anchor_test( ...
    im_path, curr_im_name, proposals_ori, use_coco, coco_gt, gt_path, ...
    catch_total, inst_total, roi_followup, cat_book)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

top_k = 300;

if roi_followup
    
    to_show_proposal{1} = proposals_ori.raw_boxes_roi{1,1};
    % PAY ATTENTION TO THIS. In 'visualize_test', we only have results of
    % individual nms value; however, in 'proposal_im_detect', we merged all
    % possible nms values (say 0.7:-0.05:0.5), that's why {5}.
    try
        to_show_proposal{2} = proposals_ori.aboxes{5};  % nms=0.5
    catch
        % choose several proposals after regression
        to_show_proposal{2} = proposals_ori.aboxes;     % nms=0.5    
        to_show_proposal = process_regression_result(to_show_proposal, proposals_ori.raw_boxes_roi);
    end
    ov_range = 0.5 : 0.05 : 0.95;
    
else
    % normal function
    to_show_proposal{1} = proposals_ori;
    ov_range = 0.5;
end

if use_coco
    addpath(genpath('./data/datasets/coco/coco_eval'));
    im = imread([im_path '/' curr_im_name '.jpg']);
else
    ld = load('./data/datasets/ilsvrc14_det/ILSVRC2014_devkit/data/meta_det.mat');
    gt_info = ld.synsets_det; clear ld;
    dataset_root = './data/datasets/ilsvrc14_det/ILSVRC2014_devkit';
    addpath([dataset_root '/evaluation']);
    im = imread([im_path '/' curr_im_name '.JPEG']);
end

%%
for sub_fig_iter = 1:length(to_show_proposal)
    
    subplot(1, length(to_show_proposal), sub_fig_iter);
    hold on;
    imshow(mat2gray(im));
    
    % draw gt
    if use_coco
        [gt_info, empty_gt] = collect_coco_gt(coco_gt, curr_im_name);
        gt_rois = gt_info.boxes;
        if empty_gt, return; end
    else
        rec = VOCreadxml([gt_path '/' curr_im_name '.xml']);
        % if length(rec.annotation.object) > 2, keyboard; end
        gt_rois = zeros(length(rec.annotation.object), 4);
        for i = 1:length(rec.annotation.object)
            gt_obj = cellfun(@(x) str2double(x), ...
                struct2cell(rec.annotation.object(i).bndbox), 'uniformoutput', false);
            gt_obj = cell2mat(gt_obj);
            gt_obj = [gt_obj(1) gt_obj(3) gt_obj(2) gt_obj(4)];
            gt_rois(i, :) = gt_obj;
        end
    end
    
    % draw boxes
    try
        proposals = floor(to_show_proposal{sub_fig_iter}(1:top_k, 1:4));
    catch
        % in case there are not enough top_k
        proposals = floor(to_show_proposal{sub_fig_iter}(:, 1:4));
    end
    proposals(end, :) = [1 1 size(im, 2), size(im, 1)];
    
    proposals = double(proposals);
    [true_overlap, ~] = compute_overlap_hyli(gt_rois, proposals);
    
    gt_obj_fail = [];
    gt_fail_label = cell(1);
    if use_coco, inst_num = size(gt_rois, 1);
    else inst_num = length(rec.annotation.object); end
    
    curr_recall = zeros(length(ov_range), 1);
    
    for ov_iter = 1 : length(ov_range)
        
        draw_fig = false;
        cnt = 0;
        curr_ov = ov_range(ov_iter);
        if curr_ov == 0.5, draw_fig = true; end
        
        for i = 1 : inst_num
            gt_obj = gt_rois(i, :);
            match_prop = proposals(true_overlap(i).overlap >= curr_ov, :);
            
            if use_coco
                lable_name = cat_book(...
                    extractfield(cat_book, 'id') == gt_info.class(i) ).name;
            else
                lable_name = gt_info(strcmp(rec.annotation.object(i).name, ...
                    extractfield(gt_info, 'WNID'))).name;
            end
            
            if ~isempty(match_prop)
                if draw_fig
                    prop_label = cell(size(match_prop, 1), 1);
                    for kk = 1:length(prop_label), prop_label{kk} = lable_name; end
                    % draw match boxes
                    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'b', ...
                        'linewidth', 2), num2cell(match_prop, 2));
%                     cellfun(@(x, y) text(x(1), x(2), sprintf('%s', y), 'backgroundcolor', 'b'), ...
%                         num2cell(match_prop, 2), prop_label);
                    % draw successfully catched up GTs
                    rectangle('Position', RectLTRB2LTWH(gt_obj), 'EdgeColor', 'r', 'linewidth', 2);
%                     text(gt_obj(1), gt_obj(2), lable_name, 'backgroundcolor', 'r');
                end
            else
                cnt = cnt + 1;
                gt_obj_fail = [gt_obj_fail; gt_obj];
                gt_fail_label{cnt, 1} = lable_name;
            end
        end
        curr_recall(ov_iter) = (inst_num - cnt) / inst_num;
        
        if draw_fig
            
            % draw failure GTs
            if ~isempty(gt_obj_fail)
                
                cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'magenta', ...
                    'linewidth', 2, 'linestyle', '--'), num2cell(gt_obj_fail, 2));
                cellfun(@(x, y) text(x(1), x(2), sprintf('%s', y), 'backgroundcolor', 'magenta'), ...
                    num2cell(gt_obj_fail, 2), gt_fail_label);
            end
            % title
            text(20, (size(im, 1)+30), sprintf('fig# %d, catch: %d / fail: %d, name: %s\n', ...
                sub_fig_iter, size(gt_rois, 1)-size(gt_obj_fail, 1), size(gt_obj_fail, 1), strrep(curr_im_name, '_', '\_')));
        end
    end  % loopr for ov_range
    
    if roi_followup
        % title
        text(20, (size(im, 1)+60), sprintf('av recall: %.3f\n', mean(curr_recall)));
        text(20, (size(im, 1)+80), sprintf('recall detail: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n', curr_recall));
    end
    hold off;
    
    catch_total = catch_total + size(gt_rois, 1)-size(gt_obj_fail, 1);
    inst_total = inst_total + size(gt_rois, 1);
end % loop for each to-show-proposal
end

