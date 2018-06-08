clear;
dataset = 'imagenet';
top_k = 300;
ov = 0.5;

coco_gt = []; gt_path = []; cat_book = [];
if strcmp(dataset, 'imagenet')
    
    im_path = './data/datasets/ilsvrc14_det/ILSVRC2013_DET_val';
    gt_path = './data/datasets/ilsvrc14_det/ILSVRC2013_DET_bbox_val';
    addpath('./data/datasets/ilsvrc14_det/ILSVRC2014_devkit/evaluation');
    addpath('./code/utils');
%     ld = load('./data/datasets/ilsvrc14_det/ILSVRC2014_devkit/data/meta_det.mat');
%     gt_info = ld.synsets_det; clear ld;
    use_coco = false;
    
elseif strcmp(dataset, 'coco')
    
    addpath(genpath('./data/datasets/coco/coco_eval'));
    im_path = './data/datasets/coco/val2014';
    coco_gt = CocoApi('./data/datasets/coco/annotations/instances_minival2014.json');
    use_coco = true;
    cat_book = coco_gt.data.categories;
end

%%
% test_split_folder = ...
%     'D02_s31_noCrop_multiLoss/ilsvrc14_val2/final_multiLoss_multiScale_new/nms_0.50/split';
% test_split_folder = ...
%     'D06_s31_hg_threeLoss/ilsvrc14_val2/iter_40000_multiLoss_multiScale/nms_0.50/split';
% test_split_folder = ...
%     'D06_s170_hg_threeLoss/ilsvrc14_val2/iter_132500_multiLoss_multiScale/nms_0.50/split';
% test_split_folder = ...
%     '/home/hongyang/Desktop/test_res_sep_29/92.x_res_D02/nms_0.50/split';
% test_split_folder = ...
%     '/home/hongyang/Desktop/test_res_sep_29_30/D06_s170/nms_0.50/split';
% test_split_folder = ...
%     'D12_s170_dense/ilsvrc14_val2/final_dense_test/nms_0.50/split';
% test_split_folder = ...
%     '/home/hongyang/Desktop/test_res_sep_29_30/D10_s31/nms_0.50/split';
% test_split_folder = ...
%     'D14b_s170_2hg/ilsvrc14_val2/iter_157500_dense_test_local_debug_4loss/nms_0.50/split';
% % coco results
% test_split_folder = ...
%     'D15a_coco/coco_val/iter_70000_dense_test/nms_0.50/split';
% % roi-followup results
% test_split_folder = ...
%     'D16a_roi/ilsvrc14_val2/iter_160000_noDense_test/nms_0.50/split';

% to show in the paper
% test_split_folder = '~/Desktop/analyze_to_paper/run_simple_forward/d16a';
test_split_folder = '~/Desktop/analyze_to_paper/run_simple_forward/d15a';

% if filled in, we will output those results in the test_folder based on
% the designated list from wish list.
which_list_folder = '';
%which_list_folder = 'D06_s170_hg_threeLoss/ilsvrc14_val2/iter_132500_multiLoss_multiScale/nms_0.50/split';

roi_followup = false; %true;
% mode = 'free_browse';
mode = 'full_examine';  % will save the results

%%
if strcmp(mode, 'full_examine')
    success_folder = fullfile(fileparts(test_split_folder), 'success');
    mkdir_if_missing(success_folder);
    fail_folder = fullfile(fileparts(test_split_folder), 'fail');
    mkdir_if_missing(fail_folder);
end

%% detemine load_box_folder and check_dir
mat_dir = dir(['./output/rpn/' test_split_folder '/*.mat']);
if size(mat_dir, 1) == 0
    mat_dir = dir([test_split_folder '/*.mat']);
    load_box_folder = test_split_folder;
else
    load_box_folder = ['./output/rpn/' test_split_folder];
end
if size(mat_dir, 1) == 0
    error('fuck');
end

if ~isempty(which_list_folder)
    wish_list_dir = dir(['./output/rpn/' which_list_folder '/*.mat']);
    check_dir = wish_list_dir;
else
    check_dir = mat_dir;
end

%%
catch_total = 0;
inst_total = 0;
for hehe = 1:length(check_dir)
    
    close all;
    if isempty(which_list_folder)
        if strcmp(mode, 'free_browse')
            curr_im_name = mat_dir(randi(length(mat_dir), 1)).name(1:end-4);
        else
            curr_im_name = mat_dir(hehe).name(1:end-4);
        end
    else
        if strcmp(mode, 'free_browse')
            curr_im_name = wish_list_dir(randi(length(wish_list_dir), 1)).name(1:end-4);
        else
            curr_im_name = wish_list_dir(hehe).name(1:end-4);
        end
    end
    
    if strcmp(mode, 'free_browse'),
        f = figure('visible', 'on');
    else
        % f = figure('visible', 'off', 'units', 'normalized', 'position', [0 0 .2 .2]);
        % f = figure('visible', 'off');
        f = figure('visible', 'on');
    end
    
    ld = load([load_box_folder '/' curr_im_name '.mat']);
    if roi_followup
        proposals_ori = ld;
    else
        proposals_ori = ld.aboxes;
%         boxes_temp = ld.raw_boxes_roi;
%         out = process_regression_result([], boxes_temp, 'naive_2_nms7');
%         proposals_ori = out{1};
    end
    [catch_total, inst_total, gt_obj_fail] = visual_anchor_test( ...
        im_path, curr_im_name, proposals_ori, use_coco, coco_gt, gt_path, ...
        catch_total, inst_total, roi_followup, cat_book);
    
    if strcmp(mode, 'full_examine')
        if isempty(gt_obj_fail)
            print(fullfile(success_folder, [curr_im_name '.pdf']), '-dpdf', '-r0');
        else
            print(fullfile(fail_folder, [curr_im_name '.png']), '-dpng', '-r0');
        end
    end
    if strcmp(mode, 'free_browse'), keyboard; end
end

if strcmp(mode, 'full_examine')
    fprintf('\naverage recall: %.3f\n', catch_total/inst_total);
end