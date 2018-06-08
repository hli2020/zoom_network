function [recall_per_cls, mean_rec, proposals] = compute_recall_ilsvrc(prop_path, top_k, info)
% prop_path can either be a specific path to split files or a merged mat
% file; if top_k is set negative, then all boxes will be evaluated.

dataset_root = './data/datasets/ilsvrc14_det/ILSVRC2014_devkit';
addpath([dataset_root '/evaluation']);
try ov = info.ov; catch, ov = 0.5; end
try skip_check = info.skip_check; catch, skip_check = false; end

if isfield(info, 'special_choice')
    if isempty(info.special_choice)
        roi_special = false;
    else
        roi_special = true;
    end
else
    roi_special = false;
end

if ~strcmp(info.name, 'coco_val')
    switch info.name
        case 'ilsvrc14_val2'
            assert(info.flip==false);
            fid = fopen([dataset_root '/data/det_lists/val2.txt'], 'r');
            annopath = [dataset_root '/../ILSVRC2013_DET_bbox_val/'];
            
        case 'val2_original'
            fid = fopen([dataset_root '/data/det_lists/val2_original.txt'], 'r');
            annopath = [dataset_root '/../ILSVRC2013_DET_bbox_val/'];
            
        case 'ilsvrc14_val1'
            fid = fopen([dataset_root '/data/det_lists/val1.txt'], 'r');
            annopath = [dataset_root '/../ILSVRC2013_DET_bbox_val/'];
            
        case 'ilsvrc14_train14'
            fid = fopen([dataset_root '/data/det_lists/train14.txt'], 'r');
            annopath = [dataset_root '/../ILSVRC2014_DET_bbox_train/'];
    end
    if fid == -1 && strcmp(info.name, 'ilsvrc14_val2')
        fid = fopen('./data/datasets/ilsvrc14_det/val2.txt');
    end
    temp = textscan(fid, '%s%s');
    test_im_list = temp{1};
    use_coco = false;
else
    test_im_list = extractfield(info.coco.data.images, 'file_name')';
    test_im_list = cellfun(@(x) x(1:end-4), test_im_list, 'uniformoutput', false);
%     %%% TEMP
%     root_dir = './data/datasets/coco';
%     info.coco_whole = CocoApi([root_dir '/annotations/instances_val2014.json']); 
%     whole_test_list = extractfield(info.coco_whole.data.images, 'file_name')';
%     whole_test_list = cellfun(@(x) x(1:end-4), whole_test_list, 'uniformoutput', false);
%     %%% TEMP
    use_coco = true;
    cat_book = info.coco.data.categories;
end

%% compute recall
% VITAL: PREPARE 'PROPOSALS'
% merge split files together
try
    info.raw_proposals;
catch
    info.raw_proposals = [];
end

if isempty(info.raw_proposals)
    
    cprintf('blue', 'loading proposals, take some time ...\n');
    if exist(prop_path, 'file') == 2
        
        ld = load(prop_path);
        try proposals = ld.aboxes; catch, end
        try proposals = ld.proposal; catch, end
        try proposals = ld.boxes; catch, end
        try proposals = ld.new_proposals; catch, end
%         new_proposals = cell(length(test_im_list), 1);
%         for i = 1:length(test_im_list)
%             new_proposals(i) = proposals(strcmp(test_im_list{i}, whole_test_list));
%         end
    elseif exist(prop_path, 'dir') == 7
        
        % check number of files
        if ~skip_check
            split_file_path = dir([prop_path '/*.mat']);
            if length(split_file_path) < length(test_im_list)
                fprintf('actual mat number vs should-be number: %d vs %d, will quite and not compute recall\n', ...
                    length(split_file_path), length(test_im_list));
                quit();
            end
        end
        proposals = cell(length(test_im_list), 1);
        for i = 1:length(test_im_list)
            % try-catch for coco case
            try
                ld = load([prop_path '/' test_im_list{i} '.mat']);
            catch
                ld = load([prop_path '/' test_im_list{i}(1:end-3) '/' test_im_list{i} '.mat']);
            end
            if roi_special
                temp = ld.raw_boxes_roi;
                out = process_regression_result([], temp, info.special_choice);
                proposals{i} = out{1};
            else
                try proposals{i} = ld.aboxes; catch, proposals{i} = ld.boxes; end
            end
        end
    end
    if info.flip, proposals = proposals(1:2:end); end
    assert(length(proposals)==length(test_im_list));
else
    proposals = info.raw_proposals;
end
% init stats
recall_per_cls = [];
if use_coco
    cat_num = 80;
    recall_per_cls(cat_num).name = 'fuck';
    for i = 1:cat_num
        recall_per_cls(i).wnid = cat_book(i).id;
        recall_per_cls(i).name = cat_book(i).name;
        recall_per_cls(i).total_inst = 0;
        recall_per_cls(i).correct_inst = 0;
        recall_per_cls(i).recall = 0;
    end
else
    cat_num = 200;
    ld = load([dataset_root '/data/meta_det.mat']);
    try synsets = ld.synsets_det; catch, synsets = ld.synsets; end
    recall_per_cls(cat_num).name = 'fuck';
    for i = 1:cat_num
        recall_per_cls(i).wnid = synsets(i).WNID;
        recall_per_cls(i).name = synsets(i).name;
        recall_per_cls(i).total_inst = 0;
        recall_per_cls(i).correct_inst = 0;
        recall_per_cls(i).recall = 0;
    end
    wnid_list = extractfield(recall_per_cls, 'wnid')';
end

%%
show_num = 3000;
for i = 1:length(test_im_list)
    
    if i == 1 || i == length(test_im_list) || mod(i, show_num)==0
        fprintf('compute recall: (%d/%d)\n', i, length(test_im_list));
    end
    % per image!
    % first collect GT boxes of this class in this image
    if use_coco
        [gt_info, empty_gt] = collect_coco_gt(info.coco, test_im_list{i});
        if empty_gt, continue; end
        gt_rois = gt_info.boxes;
        gt_cls = gt_info.class;
        try
            curr_area_range = info.scale_range.^2;
            gt_area = (gt_rois(:, 1)-gt_rois(:,3)).*(gt_rois(:, 2)-gt_rois(:,4));
            keep_inds = find(gt_area>=curr_area_range(1) & gt_area<=curr_area_range(2));
            if isempty(keep_inds)
                continue;
            else
                gt_rois = gt_info.boxes(keep_inds, :);
                gt_cls = gt_info.class(keep_inds);
            end
        catch
        end
        cls_list = unique(gt_cls);
    else
        rec = VOCreadxml([annopath, test_im_list{i}, '.xml']);
        try
            temp = squeeze(struct2cell(rec.annotation.object));
        catch
            % no object in this fucking image, pass it
            continue;
        end
        try
            curr_area_range = info.scale_range.^2;
            gt_rois = str2double(squeeze(struct2cell(cell2mat(temp(2, :)))))';
            gt_rois = gt_rois(:, [1 3 2 4]);
            gt_area = (gt_rois(:, 1)-gt_rois(:,3)).*(gt_rois(:, 2)-gt_rois(:,4));
            keep_inds = find(gt_area>=curr_area_range(1) & gt_area<=curr_area_range(2));
            if isempty(keep_inds)
                continue;
            else
                temp = temp(:, keep_inds);
            end
        catch
        end
        cls_list = unique(temp(1, :));
    end
    
    if ~use_coco
        w = str2double(rec.annotation.size.width);
        h = str2double(rec.annotation.size.height);
    end
    
    bbox_temp = proposals{i};
    if iscell(bbox_temp)
        bbox_temp = cat(1, bbox_temp{:});   % for gop method
    end
    for j = 1:length(cls_list)
        % per class!
        if ~use_coco
            wnid = cls_list{j};
            cls_id = find(strcmp(wnid, wnid_list)==1);
            % get the objects of this class in this image
            temp_ind = cellfun(@(x) strcmp(x, wnid), temp(1,:));
            objects = temp(2, temp_ind);
            gt = str2double(squeeze(struct2cell(cell2mat(objects))))';
            gt = gt(:, [1 3 2 4]);
        else
            wnid = cls_list(j);
            cls_id = find(extractfield(recall_per_cls, 'wnid')==wnid);
            gt = gt_rois(wnid==gt_cls, :);
        end
        
        if ~isempty(bbox_temp)
            try
                bbox_candidate = floor(bbox_temp(1:top_k, 1:4));
            catch
                bbox_candidate = floor(bbox_temp(:, 1:4));
            end
            if ~use_coco
                % add a result of the whole image
                bbox_candidate(end, :) = [1 1 w h];
            end
            [true_overlap, ~] = compute_overlap_hyli(gt, bbox_candidate);
            correct_inst = sum(extractfield(true_overlap, 'max') >= ov);
            
            recall_per_cls(cls_id).correct_inst = ...
                recall_per_cls(cls_id).correct_inst + correct_inst;
        end
        recall_per_cls(cls_id).total_inst = ...
            recall_per_cls(cls_id).total_inst + size(gt, 1);
    end
    
end
disp('');
correct_cnt = 0;
total_cnt = 0;
for i = 1:cat_num
    recall_per_cls(i).recall = ...
        recall_per_cls(i).correct_inst/recall_per_cls(i).total_inst;
    correct_cnt = correct_cnt + recall_per_cls(i).correct_inst;
    total_cnt = total_cnt + recall_per_cls(i).total_inst;
end

% compute mean recall here
mean_rec = 100*(correct_cnt/total_cnt);
