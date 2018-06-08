function [net_inputs, all_rois_blob, image_size] = ...
    rpn_fetch_data(conf, imdb_info_required)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

gpu_num         = length(conf.gpu_id);
assert(gpu_num == length(imdb_info_required));
net_inputs      = cell(gpu_num, 1);
all_rois_blob   = cell(gpu_num, 1);
image_size      = cell(gpu_num, 1);
empty_gt        = false;   

for i = 1:length(conf.gpu_id)
    % loop for each gpu card
    conf.temp_which_gpu = i;
    
    % check if the image has empty boxes (due to various reasons)
    if strcmp(conf.dataset, 'coco')
        % update the field 'roidb_merge' for coco
        [imdb_info_required{i}.roidb_merge, empty_gt] = ...
            collect_coco_gt(conf.COCO.cocoApi, imdb_info_required{i}.im_id);
    else
        % for imagenet, imagenet_3k
        if isempty(imdb_info_required{i}.roidb_merge.boxes), empty_gt = true; end
    end
    
    if ~empty_gt        
        if ~isempty(conf.PASS.recursive_rpn_box)
            curr_recursive_rpn_box = conf.PASS.recursive_rpn_box{i};
        else
            curr_recursive_rpn_box = [];
        end
        [net_inputs{i}, ~, ~, all_rois_blob{i}, image_size{i}] = ...
            proposal_get_minibatch(conf, imdb_info_required{i}, curr_recursive_rpn_box);
    else
        net_inputs{i} = [];
    end
end
end

