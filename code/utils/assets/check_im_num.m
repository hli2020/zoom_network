function check_im_num()
% check image number
clear; clc;
% val1, train14

val1_path = '../datasets/ilsvrc14_det/ILSVRC2014_devkit/data/det_lists/val1.txt';
val_bbox_path = '../datasets/ilsvrc14_det/ILSVRC2013_DET_bbox_val'; % no slash
addpath('../datasets/ilsvrc14_det/ILSVRC2014_devkit/evaluation');
train_path = '../datasets/ilsvrc14_det/ILSVRC2014_devkit/data/det_lists/';
train_bbox_path = '../datasets/ilsvrc14_det/ILSVRC2014_DET_bbox_train'; % no slash

fid = fopen(val1_path, 'r');
temp = textscan(fid, '%s%s');
val1_im_list = temp{1};
%[noGT_num_val1, list] = check(val1_im_list, val_bbox_path);

for i = 1:200
    fid = fopen([train_path sprintf('train_pos_%d.txt', i)], 'r');
    temp = textscan(fid, '%s');
    train_list = temp{1};
    % only train14
    train_list = train_list(cellfun(@(x) strcmp(x(1:10), 'ILSVRC2014'), train_list));
    train(i).total_per_cls = length(train_list);
    [train(i).cnt, train(i).list] = check(train_list, train_bbox_path);
end
train_im_total = sum(extractfield(train, 'total_per_cls'));
noGT_train_cnt = sum(extractfield(train, 'cnt'));

function [no_gt_cnt_val1, no_gt_list_val1] = check(val1_im_list, val_bbox_path)

no_gt_cnt_val1 = 0;
no_gt_list_val1 = cell(1);
for i = 1:length(val1_im_list)
    rec = VOCreadxml([val_bbox_path '/' val1_im_list{i} '.xml']);
    if ~isfield(rec.annotation, 'object') || isempty(rec.annotation.object)
        no_gt_cnt_val1 = no_gt_cnt_val1+1;
        no_gt_list_val1{no_gt_cnt_val1,1} = val1_im_list{i};
    end
end
