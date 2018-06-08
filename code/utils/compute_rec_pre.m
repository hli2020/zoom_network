function [mean_rec, mean_rec_size, recall_per_cls] = compute_rec_pre(imdb_info, boxes, eval)
% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2015-2017 Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------

mean_rec_size = [];
recall_per_cls = [];

% init stats
% TODO: 'recall_per_cls'
% if strcmp(imdb_info.dataset, 'voc'), cat_num = 20; end
% recall_per_cls(cat_num).name = 'init';
% recall_per_cls(i).wnid = synsets(i).WNID;
% recall_per_cls(i).name = synsets(i).name;
% recall_per_cls(i).total_inst = 0;
% recall_per_cls(i).correct_inst = 0;
% recall_per_cls(i).recall = 0;
ov_range = eval.ov;
topK_range = eval.top_k;
catch_gt_cnt = zeros(length(topK_range), length(ov_range));
gt_cnt = 0;

if eval.eval_size
    
    scale_num = length(eval.scale_name);
    scale_area = cell(scale_num, 1);
    
    for i = 1:scale_num
        if i == 1, scale_area{i} = [0, eval.scale_size(i)^2];
        elseif i == scale_num, scale_area{i} = [eval.scale_size(end)^2, inf];
        else scale_area{i} = [eval.scale_size(i-1)^2 eval.scale_size(i)^2];
        end
    end
    catch_gt_cnt_scale = zeros(scale_num, length(topK_range), length(ov_range));
    gt_cnt_scale = zeros(scale_num, 1);
end

%%
im_num = size(boxes, 1);
show_num = 3000;
for i = 1:im_num
    
    if i == 1 || i == length(im_num) || mod(i, show_num)==1
        fprintf('compute recall: (%d/%d)\n', i, length(im_num));
    end
    
    curr_boxes = boxes{i, 1};
    if iscell(curr_boxes)
        curr_boxes = cat(1, curr_boxes{:});   % for gop method
    end
    im_name = boxes{i, 2};
    index = strcmp(im_name, imdb_info.imdb_merge_path);
    gt_info = imdb_info.roidb_merge(index);
    h = imdb_info.imdb_merge_size(index, 1);
    w = imdb_info.imdb_merge_size(index, 2);
    to_eval_gt = gt_info.boxes;
    gt_cnt = gt_cnt + size(to_eval_gt, 1);
    
    if eval.eval_size
        [gt_size_ind, gt_cnt_scale] = allocate_gt_size(...
            scale_area, to_eval_gt, gt_cnt_scale);
    end
    
    to_eval_box = zeros(size(curr_boxes, 1)+1, 4);
    to_eval_box(2:end, :) = curr_boxes(:, 1:4);
    to_eval_box(1, :) = [1 1 w h];
    % matrix [gt_num x box_num]
    overlap = boxoverlap(to_eval_gt, to_eval_box);
    ov_cell = num2cell(overlap, 2);
    
    for m = 1:length(topK_range)
        for n = 1:length(ov_range)
            currTop = topK_range(m);
            currOv = ov_range(n);
            max_ov = cellfun(@(x) max(x(1:min(currTop, length(x))) >= currOv, [], 2), ov_cell);
            catch_gt_cnt(m, n) = catch_gt_cnt(m, n) + sum(max_ov);
            
            if eval.eval_size
                catch_gt_cnt_scale(:, m, n) = catch_gt_cnt_scale(:, m, n) + ...
                    tiny2(gt_size_ind, max_ov, scale_num);
            end
        end
    end
end

mean_rec = catch_gt_cnt ./ gt_cnt;
if eval.eval_size
    mean_rec_size = bsxfun(@rdivide, catch_gt_cnt_scale, gt_cnt_scale);
end
end

function [gt_size_ind, gt_cnt_scale] = allocate_gt_size(scale_area, to_eval_gt, gt_cnt_scale)
gt_area = (to_eval_gt(:, 1)-to_eval_gt(:, 3)+1).*(to_eval_gt(:, 2)-to_eval_gt(:, 4)+1);
gt_size_ind = cellfun(@(x) tiny(scale_area, x), num2cell(gt_area));

for i = 1:length(gt_cnt_scale)
    gt_cnt_scale(i) = gt_cnt_scale(i) + sum(gt_size_ind==i);
end
end

function index = tiny(scale_area, area)
for kk = 1:length(scale_area)
    if (area <= scale_area{kk}(2)) && (scale_area{kk}(1) < area)
        index = kk; break;
    end
end
end

function catch_gt_per_scale = tiny2(gt_size_ind, max_ov, scale_num)
catch_gt_per_scale = zeros(scale_num, 1);
for i = 1:scale_num
    catch_gt_per_scale(i) = sum(max_ov(gt_size_ind==i));
end
end
