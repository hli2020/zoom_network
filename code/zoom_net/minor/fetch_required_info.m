function imdb_info_required = fetch_required_info(imdb_info, sub_ind_list, dataset)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

imdb_info_required = cell(length(sub_ind_list), 1);

for i = 1:length(imdb_info_required)
        
    imdb_info_required{i}.imdb_merge_size = imdb_info.imdb_merge_size(sub_ind_list{i}, :);
    imdb_info_required{i}.im_path_root = imdb_info.im_path_root;
    imdb_info_required{i}.imdb_merge_path = imdb_info.imdb_merge_path(sub_ind_list{i});
    if strcmp(dataset, 'coco')
        imdb_info_required{i}.im_id = imdb_info.im_id(sub_ind_list{i});
        
    elseif strcmp(dataset, 'imagenet') || strcmp(dataset, 'voc')
        imdb_info_required{i}.roidb_merge = imdb_info.roidb_merge(sub_ind_list{i});
        
    elseif strcmp(dataset, 'imagenet_3k')
        % update the 'roidb_merge' field
        imdb_info_required{i}.roidb_merge.boxes = cell2mat(imdb_info.bbox(sub_ind_list{i}));
        obj_num = size(imdb_info_required{i}.roidb_merge.boxes, 1);
        imdb_info_required{i}.roidb_merge.class = ...
            repmat(imdb_info.label(sub_ind_list{i}), [obj_num 1]);
    end
end