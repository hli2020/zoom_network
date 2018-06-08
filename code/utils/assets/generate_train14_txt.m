function generate_train14_txt()
%GENERATE_TRAIN14_TXT Summary of this function goes here
%   Detailed explanation goes here
addpath('../../datasets/ilsvrc14_det/ILSVRC2014_devkit/evaluation');

path = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_DET_train/';
path_dir = dir([path 'ILSVRC2014_*']);

train_bbox_path = '../../datasets/ilsvrc14_det/ILSVRC2014_DET_bbox_train'; % no slash

% train14
fid = fopen([path '../ILSVRC2014_devkit/data/det_lists/train14.txt'], 'w');
for i = 1:length(path_dir)
    
    im_dir = dir([path path_dir(i).name '/*.JPEG']);
    for j = 1:length(im_dir)
        
        rec = VOCreadxml([train_bbox_path '/' ...
            path_dir(i).name '/' im_dir(j).name(1:end-5) '.xml']);
        if ~(~isfield(rec.annotation, 'object') || isempty(rec.annotation.object))
            fprintf(fid, [path_dir(i).name '/' im_dir(j).name(1:end-5) '\n']);
        end
    end
end


val_bbox_path = '../../datasets/ilsvrc14_det/ILSVRC2013_DET_bbox_val'; % no slash
% val1
fid = fopen([path '../ILSVRC2014_devkit/data/det_lists/val1.txt'], 'w');
fid_temp = fopen([path '../ILSVRC2014_devkit/data/det_lists/val1_original.txt'], 'r');
temp = textscan(fid_temp, '%s %d');
im_list = temp{1};

cnt = 0;
for i = 1:length(im_list)
    
    rec = VOCreadxml([val_bbox_path '/' im_list{i} '.xml']);
    if ~(~isfield(rec.annotation, 'object') || isempty(rec.annotation.object))
        cnt = cnt + 1;
        fprintf(fid, [im_list{i} ' ' sprintf('%d', cnt) '\n']);
    end
end
fclose(fid_temp);

% val2
fid = fopen([path '../ILSVRC2014_devkit/data/det_lists/val2.txt'], 'w');
fid_temp = fopen([path '../ILSVRC2014_devkit/data/det_lists/val2_original.txt'], 'r');
temp = textscan(fid_temp, '%s %d');
im_list = temp{1};

cnt = 0;
for i = 1:length(im_list)
    
    rec = VOCreadxml([val_bbox_path '/' im_list{i} '.xml']);
    if ~(~isfield(rec.annotation, 'object') || isempty(rec.annotation.object))
        cnt = cnt + 1;
        fprintf(fid, [im_list{i} ' ' sprintf('%d', cnt) '\n']);
    end
end


