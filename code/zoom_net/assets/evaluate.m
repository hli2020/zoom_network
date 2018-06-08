% plot average recall
close all; clear;
% folder_name{1} = 'coco_attract';
% input_proposals = ...
%     '../externalBox/COCO_dataset_bbox/AttractionNet/mscoco_val_2014/boxes';

% imdb.name = 'coco_val';
% folder_name{1} = 'coco_solo_no_regress';
% only_one_method = true;
% input_proposals = ...
%     './output/rpn/D15a_coco_s170_resume/coco_val/iter_50000_dense_test/nms_0.70/split';

% % for coco
% imdb.name = 'coco_val';
% root = '../externalBox/data_I_want_coco';
% special_care = false;
% folder_name{1} = 'BING';
% folder_name{2} = 'edge_boxes_70';
% folder_name{3} = 'endres';
% folder_name{4} = 'geodesic';
% folder_name{5} = 'MCG';
% folder_name{6} = 'randomized_prims';
% folder_name{7} = 'rigor';
% folder_name{8} = 'selective_search';

% imdb.name = 'coco_val';
% root = '../externalBox/data_I_want_coco';
% special_care = true;
% % folder_name{1} = 'fast_dbox_data_deepBox_coco'; file_name{1} = 'result_5k.mat'; %'val2014.mat';
% folder_name{1} = 'deepmask-coco-val-bbox'; file_name{1} = 'deepMask_final';
% folder_name{2} = 'sharpmask-coco-val-bbox'; file_name{2} = 'sharpMask_final';


% % for imagenet
imdb.name = 'ilsvrc14_val2';
% root = '../externalBox/data_I_want';
% folder_name{1} = 'split_d15a_s170_nms0.7';  % sop_baseline;
only_one_method = true;
input_proposals = './output/rpn/D16a_roi_s31/ilsvrc14_val2/iter_160000_noDense_test/nms_0.50/split';
folder_name{1} = 'sop';
special_choice = {'naive_2_nms7'};   % roi-followup

only_one_method = true;
input_proposals = '../externalBox/AttractioNet/box_proposals/author_provide/val2/oct_19_fair';
folder_name{1} = 'attract';

% folder_name{3} = 'attract';
% root = '../externalBox/data_I_want';
% folder_name{1} = 'bing';
% folder_name{2} = 'edgebox_70';
% folder_name{3} = 'endres';
% folder_name{4} = 'gop';
% folder_name{5} = 'mcg';
% folder_name{6} = 'prims';
% folder_name{7} = 'rigor';
% folder_name{8} = 'ss';

% =================================================
top_k = [10, 100, 300, 500, 700, 1000, 1500, 2000];
ov = 0.5 : 0.05 : 0.95;
imdb.flip = false;

eval_size = true;
scale_size = [32, 96];
scale_topK = 100;
scale_name = {'small', 'medium', 'large'};

if strcmp(imdb.name, 'coco_val')
    root_dir = './data/datasets/coco';
    %root_dir = 'D:/G-COCO';
    addpath(genpath([root_dir '/coco_eval']));
    imdb.image_dir  = [root_dir '/val2014'];
    imdb.extension  = 'jpg';
    imdb.coco = CocoApi([root_dir '/annotations/instances_minival2014.json']);
    dataset = 'coco';
else
    dataset = 'ilsvrc';
end

try
    only_one_method;
catch
    only_one_method = false;
end
for fuck = 1:length(folder_name) % each method
    
    save_dir = ['./experiment/evaluate/' dataset '/' folder_name{fuck}];
    mkdir_if_missing(save_dir);
    if strcmp(dataset, 'ilsvrc')
        
        if ~only_one_method, input_proposals = fullfile(root, folder_name{fuck}, 'ILSV'); end
        
    elseif strcmp(dataset, 'coco')
        
        if ~only_one_method
            if special_care
                input_proposals = fullfile(root, folder_name{fuck}, file_name{fuck});
            else
                input_proposals = fullfile(root, folder_name{fuck}, 'mat/COCO_val2014_0');
            end
            imdb.skip_check = true;
        end
    end
    
    %%
%     for mm = 1:length(special_choice)       
%         imdb.special_choice = special_choice{mm};
        
        if eval_size
            recall = zeros(length(top_k)+length(scale_size)+1, length(ov)+1);
        else
            recall = zeros(length(top_k), length(ov)+1); % plus last column: av
        end
        
        imdb.raw_proposals = [];
        try
            suffix = ['_' imdb.special_choice];
        catch
            suffix = '';
        end
        
        for i = 1:size(recall, 1)  % each top_k
            
            if i <= length(top_k)
                curr_topK = top_k(i);
                cprintf('blue', 'prop num = %d, method: %s, dataset: %s\n', ...
                    curr_topK, folder_name{fuck}, dataset);
                save_name_prefix = [imdb.name '_' sprintf('topK_%d_av_rec_', curr_topK)];
            else
                % evalute different boxes
                m = i - length(top_k);
                if m == 1
                    imdb.scale_range = [0 scale_size(m)];
                elseif m == length(scale_size)+1
                    imdb.scale_range = [scale_size(end) inf];
                else
                    imdb.scale_range = [scale_size(m-1) scale_size(m)];
                end
                curr_topK = scale_topK;
                cprintf('blue', 'area box: [%d %d], prop num = %d, method: %s, dataset: %s\n', ...
                    imdb.scale_range, curr_topK, folder_name{fuck}, dataset);
                save_name_prefix = [imdb.name '_' sprintf('%s_av_rec_', scale_name{m})];
            end
            
            check =  dir([save_dir '/' save_name_prefix '*']);
            if isempty(check)
                for j = 1:length(ov)  % each overlap
                    imdb.ov = ov(j);
                    [~, recall(i, j), proposals] = compute_recall_ilsvrc(...
                        input_proposals, curr_topK, imdb);
                    imdb.raw_proposals = proposals;
                    fprintf('recall @%.2f = %.3f,  ', ov(j), recall(i, j));
                    fprintf('%s\n', folder_name{fuck});
                end
                recall(i, length(ov)+1) = mean(recall(i, 1:length(ov)));
                cprintf('blue', 'av = %.3f\n\n', recall(i, length(ov)+1));
                save_name = sprintf('%s%.3f%s.mat', save_name_prefix, ...
                    recall(i, length(ov)+1), suffix);
                
                curr_recall = recall(i, :);
                save(fullfile(save_dir, save_name), 'curr_recall');
                
                try
                    imdb = rmfield(imdb, 'scale_range');
                catch
                end
            else
                fprintf('file exists. skip evaluation.\n');
                % TODO: reload recall results, otherwise the 'reall' will
                % be all zeros;
            end
        end
        
        recall = recall ./ 100;
        save(fullfile(save_dir, 'recall_overall.mat'), 'recall');
%     end
end
exit;
