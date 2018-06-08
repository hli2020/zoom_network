function evaluate()
% plot average recall and precision

close all; clear;
eval.top_k = [10, 100, 300, 500, 700, 1000, 1500, 2000];
eval.ov = 0.5 : 0.05 : 0.95;
eval.eval_size = true;
eval.scale_size = [32, 96];
eval.scale_name = {'small', 'medium', 'large'};

%%
dataset = 'voc';

folder_name{1} = 'F01_baseline_test';
box_source{1} = fullfile(pwd, 'output', 'F01_baseline_voc', ...
    'test', 'iter_50000', 'nms_0.60');
if strcmp(dataset, 'voc')
    imdb_info = get_voc_info('test');
end

%%
for method_iter = 1:length(folder_name) % each method
    
    save_dir = fullfile('output/evaluate', dataset, folder_name{method_iter});
    mkdir_if_missing(save_dir);
    % cell [im_num x 2], first is boxes, second is image name
    boxes = find_and_merge_box(box_source{method_iter});
    
    [recall, recall_size] = compute_rec_pre(imdb_info, boxes, eval);
    save(fullfile(save_dir, 'rec_pre.mat'), 'recall', 'recall_size');
    % TODO: draw figure
end
end

function merge_boxes = find_and_merge_box(box_folder, box_name)

if nargin < 2, box_name = 'aboxes'; end

box_file = dir([box_folder '/*.mat']);
box_file = extractfield(box_file, 'name')';
ind = cellfun(@(x) strcmp(x(1:5), 'merge'), box_file);
box_file = box_file(~ind);

if exist(fullfile(box_folder, sprintf('merge_%d.mat', length(box_file))), 'file')
    ld = load(fullfile(box_folder, sprintf('merge_%d.mat', length(box_file))));
    merge_boxes = ld.merge_boxes;
else
    merge_boxes = cell(length(box_file), 2);
    for i = 1:length(box_file)
        ld = load(fullfile(box_folder, box_file{i}));
        merge_boxes{i, 1} = ld.(box_name);
        merge_boxes{i, 2} = box_file{i}(1:end-4);
    end
    save(fullfile(box_folder, sprintf('merge_%d.mat', length(box_file))), 'merge_boxes');
end
end
