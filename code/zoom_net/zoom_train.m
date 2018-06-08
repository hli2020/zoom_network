function zoom_train(varargin)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%
%   Developver log
%       1. Jan 18: remove 'parfeval', for older version, check master
%       branch. Also clean some redundancy in the following.
%       2. Jan 18: resume mechanism currently disabled.
%
ip = inputParser;
ip.addParameter('conf',             struct(),   @isstruct);
ip.addParameter('solver',           '',         @isstr);
ip.addParameter('init_net_file',    '',         @isstr);
ip.addParameter('solverstate',      '',         @isstr);    % resume
ip.parse(varargin{:});
opts = ip.Results;

conf                                = opts.conf;
debug                               = conf.debug;
draw_boxes                          = conf.draw_boxes;
level_num                           = length(conf.anchors);
loss_bbox_weight                    = conf.loss_bbox_weight;
loss_cls_weight                     = conf.loss_cls_weight;
snapshot_interval                   = conf.save_interval;

assert(level_num == length(conf.rpn_feat_stride));
assert(level_num == length(conf.output_height_map));
assert(level_num == length(loss_bbox_weight));
assert(length(loss_bbox_weight) == length(loss_cls_weight));
% DATASET: collect imdb info
[imdb_info, bbox_means, bbox_stds, conf] = data_loader(conf);
% DRAW_BOXES: will loop around if true
if draw_boxes, draw_boxes_sub_fun(conf, imdb_info); end

prefix = repmat('%.2f ', [1 length(loss_bbox_weight)]);
cprintf('blue', sprintf('loss bbox weight is %s\tmust coincide with those in prototxts.\n', ...
    prefix), loss_bbox_weight);
cprintf('blue', sprintf('loss cls weight is %s\t\tmust coincide with those in prototxts.\n', ...
    prefix), loss_cls_weight);

%% init
cache_dir = fullfile(pwd, 'output', conf.model_id, 'train');
mkdir_if_missing(cache_dir);
% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe_solver = init_caffe(cache_dir, conf, opts);

%% MULTI-GPU TRAINING
shuffled_inds_train     = [];
train_results           = [];
iter_                   = caffe_solver.iter();
max_iter                = caffe_solver.max_iter();
start_iter              = iter_;
last_iter_input         = [];
progress_interval       = 100;
last_loss_file          = [];
last_fig_folder         = [];
train_T = conf.train_T;

cprintf('blue', 'train_T: %d\n', train_T);
if train_T > 1
    rois_per_iter = conf.rois_per_iter;
    recursive_train = true; display = sprintf('recursive T=%d', train_T);
else
    recursive_train = false; display = 'non-recursive';
end

%%
th = tic;
new_to_show_iter = 0;
% 'iter_' is the actual iteration
if debug, progress_interval = 1; end
while (iter_ < max_iter)
    
    if debug
        find_nan_weight(caffe_solver, iter_+1); tic;
    else
        find_nan_weight(caffe_solver, iter_+1);
    end
    new_to_show_iter = new_to_show_iter + 1;
    to_show_iter = new_to_show_iter;
    % to_show_iter = floor(iter_ / train_T) + 1;
    if debug, fprintf('\n== iter %d == %s\n', to_show_iter, display); end
    
    [shuffled_inds_train, sub_ind_list] = generate_random_minibatch(...
        shuffled_inds_train, imdb_info.imdb_merge_size, ...
        conf.ims_per_batch, length(conf.gpu_id));
    imdb_info_required = fetch_required_info(imdb_info, sub_ind_list, conf.dataset);
    
    conf.PASS.recursive_rpn_box = [];
    skip_recursive = false;
    for reg_iter = 1:train_T
        
        % Fetch new data for this iter
        % VITAL IMPORTANT: NEW INPUT RPN BOXES ARE PASSED VIA 'CONF.RPN_PARAM'
        [net_inputs, all_rois_blob, im_size] = rpn_fetch_data(conf, imdb_info_required);
        
        if debug && ~recursive_train, t = toc; fprintf('iter %d, data, %f\n', iter_+1, t); tic; end
        
        % some images dont have GT and we assign the specific 'net_inputs' in the
        % last iteration. Assuming that the first iter is OK.
        if any(cellfun('isempty', net_inputs))
            net_inputs(cellfun('isempty', net_inputs)) = last_iter_input(cellfun('isempty', net_inputs));
            skip_recursive = true; % in such case, we wont do recursive regression
            fprintf('\t\t== iter %d ==, skip recursive regression\n', to_show_iter);
        end
        last_iter_input = net_inputs;
        
        if debug && recursive_train
            for input_iter = 1:length(net_inputs)
                rois_num = size(net_inputs{input_iter}{2}, 4);
                pos_num_func = @(x) sum(net_inputs{input_iter}{x}==1);
                fprintf('T=%d, gpu=%d, rois_num=%d, pos_num::  %d\t%d\t%d\n', ...
                    reg_iter, input_iter, rois_num, pos_num_func(3), pos_num_func(7), pos_num_func(11));
            end
        end
        
        caffe_solver.reshape_as_input(net_inputs);
        caffe_solver.set_input_data(net_inputs);
        caffe_solver.step(1);
        iter_ = caffe_solver.iter();
        
        if skip_recursive, break; end
        if reg_iter ~= train_T && train_T > 1
            conf.PASS.recursive_rpn_box = ...
                update_rpn_boxes(caffe_solver, level_num, net_inputs, ...
                all_rois_blob, rois_per_iter, im_size);
        end
    end
    
    if debug && ~recursive_train, t = toc; fprintf('iter %d, train, %f\n', iter_, t); end
    train_results = parse_rst(train_results, caffe_solver.get_output(), iter_, debug);
    
    % report time/training progress
    if ~mod(to_show_iter, progress_interval)
        cprintf('blue', 'iter:\t%d\n', to_show_iter); time = toc(th);
        fprintf('\tTIME:\t\toneIterTime: %.2f s,\test.LeftTime: %.2f hours\n', ...
            time/progress_interval, (time/3600)*(max_iter-iter_)/progress_interval);
        show_progress(level_num, train_results, loss_bbox_weight, loss_cls_weight); th = tic;
    end
    
    % mean(caffe_solver.nets{1}.params('inception_5a/double_3x3_reduce_bn', 4).get_data)
    % size(net_inputs{1}{1})
    % save model and draw loss
    if ~mod(to_show_iter, snapshot_interval)
        snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, to_show_iter, iter_);
        if ~isempty(last_loss_file), delete(last_loss_file); end
        last_loss_file = [cache_dir '/' sprintf('loss_res_iter_%d_%d.mat', start_iter, to_show_iter)];
        save(last_loss_file, 'train_results');
        
        % draw the figure and save it!
        if ~isempty(last_fig_folder), rmdir(last_fig_folder, 's'); end
        last_fig_folder = [cache_dir '/' sprintf('loss_fig_iter_%d_%d', start_iter, to_show_iter)];
        mkdir_if_missing(last_fig_folder); close all;
        draw_loss_fig(last_fig_folder, train_results, loss_bbox_weight, loss_cls_weight, level_num);
    end
    
end

% final snapshot
snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', to_show_iter));
save([cache_dir '/' sprintf('loss_res_%d.mat', to_show_iter)], 'train_results');
snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

% diary off;
caffe.reset_all();
rng(prev_rng);
end

function caffe_solver = init_caffe(cache_dir, conf, opts)
% init caffe log
mkdir_if_missing([cache_dir '/caffe_log']);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log/train_');
caffe.init_log(caffe_log_file_base);
caffe.set_random_seed(conf.rng_seed);
% init caffe solver, copy weight from pre-trained model
solver_path = fullfile(pwd, 'model', 'zoom', [opts.solver '.prototxt']);
if conf.roi_followup
    % first half from D15a, roi part from ResNet
    caffe_solver = append_roi_weight(solver_path, conf.gpu_id, opts.init_net_file);
else
    caffe_solver = caffe.Solver(solver_path, conf.gpu_id);
    if isempty(opts.solverstate)
        caffe_solver.use_caffemodel(opts.init_net_file);
        temp = fileparts(fileparts(fileparts(opts.init_net_file)));
        file_name = strrep(opts.init_net_file, temp, '');
        fprintf('\nFinetuning from model (%s) ...\n', file_name);
        if conf.init_mirror_layer
            [~, model_name] = fileparts(opts.init_net_file);
            if strcmp(model_name, 'bn_inception_iter_900000.caffemodel') ...
                    || conf.check_certain_scale % means hg structure
                caffe_solver = copy_weights_to_mirror_layer(caffe_solver, length(conf.gpu_id));
            end
        end
    else
        % loading solverstate, resume mechanism
        % in resume we generate a different series of samples
        conf.rng_seed = 2;
        solverstate_file = fullfile(cache_dir, sprintf('%s.solverstate', opts.solverstate));
        caffemodel_file = strrep(solverstate_file, 'solverstate', 'caffemodel');
        true_caffemodel = strrep(solverstate_file, '.solverstate', '');
        cmd = sprintf('cp %s %s', caffemodel_file, true_caffemodel);
        system(cmd);
        caffe_solver.restore(true_caffemodel);
        fprintf('\nRestoring from iter %d...\n', caffe_solver.iter());
    end
end
caffe_solver.set_phase('train');
end

function [imdb_info, bbox_means, bbox_stds, conf] = data_loader(conf)

if strcmp(conf.dataset, 'imagenet')
    
    % collect std and means of bbox regression target
    imdb_info = collect_db_info(imdbs, roidbs);
    conf = collect_bbox_stats(opts, conf);
    bbox_means = conf.rpn_param.bbox_means;
    bbox_stds = conf.rpn_param.bbox_stds;
    
elseif strcmp(conf.dataset, 'coco')
    
    % don't do means and stds in coco (performance effect NOT ensured)
    imdb_info = get_coco_info(imdbs);
    imdb_info.im_path_root = conf.COCO.train_root;
    conf.COCO.cocoApi = imdbs;
    bbox_means = [0 0 0 0]; bbox_stds = [1 1 1 1];
    
elseif strcmp(conf.dataset, 'imagenet_3k')
    
    imdb_info = get_imagenet_3k_info(conf.EI);
    bbox_means = [0 0 0 0]; bbox_stds = [1 1 1 1];
    
elseif strcmp(conf.dataset, 'voc')
    imdb_info = get_voc_info(conf.mode);
    bbox_means = [0 0 0 0]; bbox_stds = [1 1 1 1];
else
    error('unknown dataset: %s', conf.dataset);
end
end

function draw_loss_fig(last_fig_folder, train_results, loss_bbox_weight, loss_cls_weight, level_num)

accu_allLevel = zeros(length(train_results.accuracy1.data), 1);
loss_cls_allLevel = zeros(size(accu_allLevel));
loss_bbox_allLevel = zeros(size(accu_allLevel));

for i = 1:level_num
    accu_name = sprintf('accuracy%d', i);
    rolling_mean(train_results.(accu_name).data);
    accu_allLevel = accu_allLevel + train_results.(accu_name).data;
    title(strrep(accu_name, '_', '\_')); print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, accu_name));
    
    loss_cls_name = sprintf('loss_cls%d', i);
    rolling_mean(loss_cls_weight(i)*train_results.(loss_cls_name).data);
    loss_cls_allLevel = loss_cls_allLevel + loss_cls_weight(i)*train_results.(loss_cls_name).data;
    title(strrep(loss_cls_name, '_', '\_')); print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, loss_cls_name));
    
    loss_bbox_name = sprintf('loss_bbox%d', i);
    rolling_mean(loss_bbox_weight(i)*train_results.(loss_bbox_name).data);
    loss_bbox_allLevel = loss_bbox_allLevel + loss_bbox_weight(i)*train_results.(loss_bbox_name).data;
    title(strrep(loss_bbox_name, '_', '\_')); print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, loss_bbox_name));
end

if level_num > 1
    loss_bbox_allLevel(loss_bbox_allLevel>=8) = 8;
    loss_cls_allLevel(loss_cls_allLevel>=8) = 8;
    rolling_mean(accu_allLevel); title('accuracy\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'accuracy_allLevel'));
    
    rolling_mean(loss_cls_allLevel); title('loss\_cls\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'loss_cls_allLevel'));
    
    rolling_mean(loss_bbox_allLevel); title('loss\_bbox\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'loss_bbox_allLevel'));
    
    rolling_mean(loss_bbox_allLevel+loss_cls_allLevel); title('loss\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'loss_allLevel'));
end

if length(fieldnames(train_results)) == 18
    
    % means intermediate supervision is added
    accu_allLevel = zeros(length(train_results.accuracy1.data), 1);
    loss_cls_allLevel = zeros(size(accu_allLevel));
    loss_bbox_allLevel = zeros(size(accu_allLevel));
    
    for i = 1:level_num
        accu_name = sprintf('accuracy%d_mi', i);
        rolling_mean(train_results.(accu_name).data);
        accu_allLevel = accu_allLevel + train_results.(accu_name).data;
        title(strrep(accu_name, '_', '\_'));
        print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, sprintf('MI_accuracy%d', i)));
        
        loss_cls_name = sprintf('loss_cls%d_mi', i);
        rolling_mean(loss_cls_weight(i)*train_results.(loss_cls_name).data);
        loss_cls_allLevel = loss_cls_allLevel + loss_cls_weight(i)*train_results.(loss_cls_name).data;
        title(strrep(loss_cls_name, '_', '\_'));
        print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, sprintf('MI_loss_cls%d', i)));
        
        loss_bbox_name = sprintf('loss_bbox%d_mi', i);
        rolling_mean(loss_bbox_weight(i)*train_results.(loss_bbox_name).data);
        loss_bbox_allLevel = loss_bbox_allLevel + loss_bbox_weight(i)*train_results.(loss_bbox_name).data;
        title(strrep(loss_bbox_name, '_', '\_'));
        print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, sprintf('MI_loss_bbox%d', i)));
    end
    
    loss_bbox_allLevel(loss_bbox_allLevel>=8) = 8;
    loss_cls_allLevel(loss_cls_allLevel>=8) = 8;
    rolling_mean(accu_allLevel); title('MI\_accuracy\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'MI_accuracy_allLevel'));
    
    rolling_mean(loss_cls_allLevel); title('MI\_loss\_cls\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'MI_loss_cls_allLevel'));
    
    rolling_mean(loss_bbox_allLevel); title('MI\_loss\_bbox\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'MI_loss_bbox_allLevel'));
    
    rolling_mean(loss_bbox_allLevel+loss_cls_allLevel); title('MI\_loss\_allLevel');
    print(gcf, '-dpng', '-r0', sprintf('%s/%s.png', last_fig_folder, 'MI_loss_allLevel'));
end
end

function accuracy = show_progress(level_num, train_results, loss_bbox_weight, loss_cls_weight)
cls_loss = zeros(level_num, 1);
bbox_loss = zeros(level_num, 1);
accuracy = zeros(level_num, 1);

for kk = 1 : level_num
    cls_loss_name = sprintf('loss_cls%d', kk);
    cls_loss(kk) = loss_cls_weight(kk)*train_results.(cls_loss_name).data(end);
    
    bbox_loss_name = sprintf('loss_bbox%d', kk);
    bbox_loss(kk) = loss_bbox_weight(kk)*train_results.(bbox_loss_name).data(end);
    
    accuracy_name = sprintf('accuracy%d', kk);
    accuracy(kk) = train_results.(accuracy_name).data(end);
    
    fprintf('\tLEVEL %d:\taccuracy: %.4f,\tloss: %.4f,\t(cls: %.4f,\tbbox: %.4f)\n', ...
        kk, accuracy(kk), (bbox_loss(kk) + cls_loss(kk)), cls_loss(kk), bbox_loss(kk));
end
if level_num > 1
    fprintf('\tTOTAL:\t\taccuracy: %.4f,\tloss: %.4f,\t(cls: %.4f,\tbbox: %.4f)\n', ...
        sum(accuracy), (sum(bbox_loss) + sum(cls_loss)), sum(cls_loss), sum(bbox_loss));
end
if length(fieldnames(train_results)) == 18
    % means intermediate supervision is added
    cls_loss = zeros(level_num, 1);
    bbox_loss = zeros(level_num, 1);
    accuracy = zeros(level_num, 1);
    
    for kk = 1 : level_num
        cls_loss_name = sprintf('loss_cls%d_mi', kk);
        cls_loss(kk) = loss_cls_weight(kk)*train_results.(cls_loss_name).data(end);
        
        bbox_loss_name = sprintf('loss_bbox%d_mi', kk);
        bbox_loss(kk) = loss_bbox_weight(kk)*train_results.(bbox_loss_name).data(end);
        
        accuracy_name = sprintf('accuracy%d_mi', kk);
        accuracy(kk) = train_results.(accuracy_name).data(end);
        
        fprintf('\tMI LEVEL %d:\taccuracy: %.4f,\tloss: %.4f,\t(cls: %.4f,\tbbox: %.4f)\n', ...
            kk, accuracy(kk), (bbox_loss(kk) + cls_loss(kk)), cls_loss(kk), bbox_loss(kk));
    end
    fprintf('\tMI TOTAL:\taccuracy: %.4f,\tloss: %.4f,\t(cls: %.4f,\tbbox: %.4f)\n', ...
        sum(accuracy), (sum(bbox_loss) + sum(cls_loss)), sum(cls_loss), sum(bbox_loss));
end

end

function [shuffled_inds, sub_inds] = generate_random_minibatch(...
    shuffled_inds, imdb_merge_size, ims_per_batch, total_num)

if isempty(shuffled_inds)
    % make sure each minibatch, only has horizontal images or vertical
    % images, to save gpu memory
    
    hori_image_inds = imdb_merge_size(:, 2) >= imdb_merge_size(:, 1);
    vert_image_inds = ~hori_image_inds;
    hori_image_inds = find(hori_image_inds);
    vert_image_inds = find(vert_image_inds);
    
    % random perm
    lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
    hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
    lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
    vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
    
    % combine sample for each ims_per_batch
    hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
    vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
    
    shuffled_inds = [hori_image_inds, vert_image_inds];
    shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
    
    shuffled_inds = num2cell(shuffled_inds, 1);
end

if nargin < 4, total_num = 1; end
if nargout > 1
    % generate minibatch training data
    try
        sub_inds = shuffled_inds(1:total_num);
        shuffled_inds(1:total_num) = [];
    catch
        % that being said, the rest few samples is less than the number of GPUs.
        sub_inds = [shuffled_inds repmat(shuffled_inds(1), [1 (total_num-length(shuffled_inds))])];
        shuffled_inds = [];
    end
end
end

function model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, iter, abs_iter)
model_path = [];
% update: if the intermediate supervsion is added, we won't save its
% paramters. TODO task here.
% bbox_stds, bbox_means: [1 x 4] double
% save the intermediate result
level_num = length(conf.anchors);
weights_back = cell(level_num, 1);
biases_back = cell(level_num, 1);
if nargin < 7
    abs_iter = conf.train_T * iter;
end

if conf.roi_followup
    layer_name_template = @(x) sprintf('roi_bbox_pred%d', x);
else
    try
        % for stack2, finetune use
        conf.snapshot_new_name;
        layer_name_template = @(x) sprintf('proposal_bbox_pred%d_new', x);
    catch
        layer_name_template = @(x) sprintf('proposal_bbox_pred%d', x);
    end
end

% merge bbox_means, bbox_stds into the model
for kk = 1:level_num
    
    bbox_pred_layer_name = layer_name_template(kk);
    weights = caffe_solver.nets{1}.params(bbox_pred_layer_name, 1).get_data();
    biases = caffe_solver.nets{1}.params(bbox_pred_layer_name, 2).get_data();
    weights_back{kk} = weights;
    biases_back{kk} = biases;
    
    if conf.roi_followup
        weights(:, 5:8) = bsxfun(@times, weights(:, 5:8), bbox_stds);
        biases(5:8) = biases(5:8) .* bbox_stds' + bbox_means';
    else
        anchor_size = size(conf.anchors{kk}, 1);
        bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
        bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);
        % seems to have some problem here. not sure how it works.
        weights = ...
            bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds;
        biases = ...
            biases .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;
    end
    caffe_solver.nets{1}.set_params_data(bbox_pred_layer_name, 1, weights);
    caffe_solver.nets{1}.set_params_data(bbox_pred_layer_name, 2, biases);
end
%% save the model
if ~ischar(iter)
    suffix = sprintf('iter_%d', iter);
    % abs_suffix = sprintf('iter_%d', abs_iter);
else
    % when iter is a string, like 'final'
    suffix = sprintf('%s', iter);
end
% model_path = [fullfile(cache_dir, suffix) '.caffemodel'];
% caffe_solver.savestate(model_path);
% [~, model_name, ext] = fileparts(model_path);
% fprintf('\nSaved as %s\n', [model_name ext]);
%
% if ~ischar(iter)
%     % move .solverstate to the folder
%     % [~, solver_name, solver_ext] = fileparts(solverstate_file);
%     solverstate_file = [fullfile(cache_dir, suffix) '.solverstate'];
%     solverstate_file_abs = sprintf('./_%s.solverstate', abs_suffix);
%     cmd = sprintf('mv %s %s', solverstate_file_abs, solverstate_file);
%     system(cmd);
%     fprintf('Saved as %s\n', [model_name '.solverstate']);
% end
caffe_solver.savestate(fullfile(cache_dir, suffix));
% excute the following if using 'CaffeMex_v2'
cmd = sprintf('mv %s %s', ...
    fullfile(cache_dir, suffix), ...
    fullfile(cache_dir, [suffix '.caffemodel']));
system(cmd);

%% restore net to original state
for kk = 1:level_num
    bbox_pred_layer_name = layer_name_template(kk);
    caffe_solver.nets{1}.set_params_data(bbox_pred_layer_name, 1, weights_back{kk});
    caffe_solver.nets{1}.set_params_data(bbox_pred_layer_name, 2, biases_back{kk});
end
end

function draw_boxes_sub_fun(conf, imdb_info)

while true
    
    close all;
    sub_ind_list{1} = randperm(length(imdb_info.imdb_merge_size), 1);
    % sub_ind_list{1} = 15808;
    % sub_ind_list{1} = 6422; %8344; 6422; 5659; %79591; %72672; %13312;
    imdb_info_required = fetch_required_info(imdb_info, sub_ind_list, conf.dataset);
    if strcmp(conf.dataset, 'coco')
        imdb_info_required{1}.roidb_merge = ...
            collect_coco_gt(conf.cocoApi, imdb_info_required{1}.im_id);
    end
    fprintf('random index is %d\n', sub_ind_list{1});
    conf.temp_which_ind = sub_ind_list{1};
    net_input_single_card = proposal_get_minibatch(conf, imdb_info_required{1});
end
end
