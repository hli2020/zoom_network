clear;

% root = '../externalBox/data_I_want_coco/deepmask-coco-val-bbox/deepMask';
root = '../externalBox/data_I_want_coco/sharpmask-coco-val-bbox/sharpMask2';
save_path = [root '/../sharpMask_final'];
file_dir = dir([root '/*.json']);

addpath(genpath('./data/datasets/coco/coco_eval'));
coco = CocoApi('./data/datasets/coco/annotations/instances_minival2014.json');
test_im_list = extractfield(coco.data.images, 'file_name')';
test_im_list = cellfun(@(x) x(1:end-4), test_im_list, 'uniformoutput', false);
test_im_id = extractfield(coco.data.images, 'id')';

content = cell(length(test_im_list), 1);
%%
log_down_useful_info = cell(length(file_dir), 1);
parfor i = 1:length(file_dir)
    
    curr_file = loadjson([root '/' file_dir(i).name], ...
        'SimplifyCell', 1, 'FastArrayParser', 1, 'showprogress', 0)';
    
    im_id_list = extractfield(curr_file, 'image_id')';
    im_id_pool = unique(im_id_list);
    
    cnt = 0;
    curr_file_useful_info = cell(1);
    for j = 1:length(im_id_pool)
        
        if sum(im_id_pool(j)==test_im_id) > 0
            cnt = cnt+1;
            curr_file_useful_info{cnt,1} = curr_file(im_id_list == im_id_pool(j));
        end
    end
    log_down_useful_info{i} = curr_file_useful_info;
end

%%
all_useful = cat(1, log_down_useful_info{:});
for i = 1:length(all_useful)
    
    curr_struct = all_useful{i};
    curr_im_id = curr_struct(1).image_id;
    boxes = reshape(extractfield(curr_struct, 'bbox'), 4, [])';
    boxes = [boxes(:, 1) boxes(:, 2) ...
        (boxes(:, 1)+boxes(:, 3)) (boxes(:, 2)+boxes(:, 4))];
    scores = extractfield(curr_struct, 'score')';
    
    content{curr_im_id==test_im_id} = cat(1, ...
        content{curr_im_id==test_im_id}, [boxes, scores]);
end

%% save
mkdir_if_missing(save_path);
for i = 1:length(content)
    boxes = content{i};
    if isempty(boxes)
        error('%d is empty\n', i);
    end
    boxes = boxes(sort(boxes(:, end), 'descend'), :);
    save([save_path '/' test_im_list{i} '.mat'], 'boxes');
end