% merge imagenet train/val of D16a

which_set = 'val'; %'train'; % 'val';

ld = load('./data/imdb/train_val_list.mat');
if strcmp(which_set, 'train')
    
    image_ids = cellfun(@(x) x(21:end-5), ...
        ld.train_list, 'uniformoutput', false);
    save_name = 'solo_regress_train.mat';
    
elseif strcmp(which_set, 'val')
    
    image_ids = cellfun(@(x) x(21:end-5), ...
        ld.val_list, 'uniformoutput', false);
    save_name = 'solo_regress_val.mat';
end

total_num = length(image_ids);
box_result(total_num).name = '';
box_result(total_num).box = [];
top_k = 1000;
root = './output/rpn/D16a_roi_s31';

non_exist_list = [];
cnt = 0;
val_search_space{1} = 'ilsvrc14_val2';
val_search_space{2} = 'train_liuyu_addition';
val_search_space{3} = 'val_liuyu_addition';

str_template = @(x) sprintf('%s/iter_160000_noDense_test/nms_0.50/split', x);

%%
% first things first, check if all files exist!
for i = 1:total_num
    
    curr_name = image_ids{i};
    
    if curr_name(1) == 't', curr_name = [curr_name(7:end) '.mat']; end
    if curr_name(1) == 'v', curr_name = [curr_name(5:end) '.mat']; end
    
    if strcmp(curr_name(1:16), 'ILSVRC2013_train')
        
        file_name = fullfile(root, str_template('train_liuyu_addition'), curr_name);
        if ~exist(file_name, 'file'), cnt = cnt+1; non_exist_list{cnt} = curr_name; end
        
    elseif strcmp(curr_name(1:16), 'ILSVRC2014_train')
        
        file_name = fullfile(root, str_template('ilsvrc14_train14'), curr_name);
        if ~exist(file_name, 'file'), cnt = cnt+1; non_exist_list{cnt} = curr_name; end
        
        % could be train/val
    elseif strcmp(curr_name(1:14), 'ILSVRC2012_val') || strcmp(curr_name(1:14), 'ILSVRC2013_val')
        
        find_it = false;
        for kk = 1:length(val_search_space)
            file_name = fullfile(root, str_template(val_search_space{kk}), curr_name);
            if exist(file_name, 'file'), find_it = true; break; end
        end
        if ~find_it, cnt = cnt+1; non_exist_list{cnt} = curr_name; end
    else
        error('unknown name format');
    end
end

if ~isempty(non_exist_list)
    %disp(non_exist_list);
    fprintf('find %d non-existant files!\n', length(non_exist_list));
    fprintf('will ignore them and merge now...\n');
    %keyboard;
else
    disp('all files exist! begin to merge them!');
end

%%
for i = 1:total_num
    
    origin_name = image_ids{i};
    curr_name = origin_name;
    
    if curr_name(1) == 't', curr_name = [curr_name(7:end) '.mat']; end
    if curr_name(1) == 'v', curr_name = [curr_name(5:end) '.mat']; end
    
    if strcmp(curr_name(1:16), 'ILSVRC2013_train')
        
        file_name = fullfile(root, str_template('train_liuyu_addition'), curr_name);
        
    elseif strcmp(curr_name(1:16), 'ILSVRC2014_train')
        
        file_name = fullfile(root, str_template('ilsvrc14_train14'), curr_name);
        
        % could be train/val
    elseif strcmp(curr_name(1:14), 'ILSVRC2012_val') || strcmp(curr_name(1:14), 'ILSVRC2013_val')
        
        for kk = 1:length(val_search_space)
            file_name = fullfile(root, str_template(val_search_space{kk}), curr_name);
            if exist(file_name, 'file'), break; end
        end
    end
    
    try
        ld = load(file_name);
        try
            boxes_temp = ld.raw_boxes_roi;
            out = process_regression_result([], boxes_temp, 'naive_2_nms7');
            boxes = out{1};
        catch
            boxes = ld.aboxes;
        end
    catch
        boxes = [1 1 10 10];
    end
    box_result(i).name = [origin_name '.mat'];
    box_result(i).box = single(boxes(1:min(top_k, size(boxes, 1)), 1:4));
end

save(save_name, 'box_result', '-v7.3');
exit();
%==========================================================
% %mat_file = dir('./train_*.mat');
% mat_file = dir('./val_*.mat');
%
% %total_num = 175129;
% total_num = 9917;
% box_result(total_num).name = '';
% box_result(total_num).box = [];
% check_ = zeros(total_num, 1);
%
% for i = 1:length(mat_file)
%
%     ld = load(['./' mat_file(i).name]);
%     curr_box = ld.box_result;
%     temp = sscanf(mat_file(i).name, 'val_ck%d_absInd_%d_%d_total%d.mat');
%     %temp = sscanf(mat_file(i).name, 'train_ck%d_absInd_%d_%d_total%d.mat');
%     start_id = temp(2);
%     end_id = temp(3);
%     box_result(start_id:end_id) = curr_box;
%     check_(start_id:end_id) = 1;
% end
%
% if ~all(check_)
%     warning('some entries are empty!');
% end
%
% %save('./ss_train.mat', 'box_result', '-v7.3');
% save('./ss_val.mat', 'box_result', '-v7.3');
