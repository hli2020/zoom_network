function [conf, dataset] = default_config(mode, conf)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

if strcmp(mode, 'train')
    
    conf.mode                                       = 'train';
    conf.gpu_id                                     = 0:3;
    conf.dynamic_train_scale                        = true;
    conf.batch_size                                 = 256;
    conf.ims_per_batch                              = 1;
    conf.rpn_max_size                               = 900;
    conf.rpn_min_size                               = 100;
    % make samples (fg/bg) more balanced
    conf.adjust_sample_ratio                        = true;
    conf.fg_fraction                                = 0.5;
    conf.fg_thresh                                  = 0.6;
    conf.bg_thresh_hi                               = 0.15;
    conf.bg_thresh_lo                               = eps;
    conf.use_flip                                   = false;
    % whether use other proposals as training data
    conf.target_only_gt                             = true;
    conf.loss_bbox_weight                           = [1 1 1];
    conf.loss_cls_weight                            = [1 1 1];
    % training logistics
    conf.save_interval                              = 5000;
    conf.init_net_file                              = fullfile(pwd, ...
        'data', 'bn_inception_iter_900000.caffemodel');
    
elseif strcmp(mode, 'test')
    
    conf.mode                                       = 'test';
    conf.gpu_id                                     = 0;
    conf.test.T                                     = 1;
    conf.test.nms_mode                              = 'normal_nms';
    conf.test.nms_thres                             = 0.6;  % 0.7 : -0.05 : 0.5;
    conf.test.box_vote                              = false;
    conf.test.box_vote_ov_thres                     = 0.8;
    
    conf.test.keepTopBeforeNMS                      = 15000;
    conf.test.nms_keep_at_level                     = 3000;
    conf.test_scale_range                           = (1400:-50:250);
    conf.test_multiscale_max_size                   = 1600;
    % filter out small boxes during test
    conf.test.min_box_size                          = 2;
    conf.test.eval_top_k                            = 300;

    % not sure if they are active now
    conf.test.rois_per_iter = [];
    conf.test.useTopBox_num = [];
end

%% configuration
conf.rng_seed                                       = 4;
% different depth, corresponding to various resolutions
conf.multi_depth                                    = true;
% has numel(rpn_feat_stride) depths in the network
conf.rpn_feat_stride                                = [8, 16, 32];
conf.revise_boundary_box                            = true;
% only some scales are feasible as input due to the network structure
conf.check_certain_scale                            = true;
% add a RoI head
conf.roi_followup                                   = false;
conf.train_T                                        = 1;
% class number (fg/bg/intermediate)
conf.add_gray_cls                                   = true;
conf.gray_hi                                        = 0.49;
conf.gray_lo                                        = 0.25;
% anchor
conf.base_size                                      = 16*sqrt(2);
conf.anchor_scale                                   = 2.^(0:5);
conf.ratios                                         = [0.25, 0.5, 1, 2, 3];
% preset possible image size
conf.min_size                                       = max(conf.rpn_feat_stride);
conf.max_size                                       = 2000;

% || image mean and init network (will be passed on to variable 'model' outside)
ld = load('data/image_mean.mat');
if strcmp(conf.dataset, 'coco')
    conf.DATA.image_mean = ld.coco;
else
    % strcmp(conf.dataset, 'imagenet') || strcmp(conf.dataset, 'imagenet_3k')
    conf.DATA.image_mean = ld.ilsvrc14_det; 
end
clear ld;

%% dataset
dataset = [];
if strcmp(conf.dataset, 'coco')
    
    root_dir = './data/datasets/coco';
    addpath(genpath([root_dir '/coco_eval']));
    cprintf('blue', 'loading COCO dataset ... \n');
    dataset.imdb_train{1} = CocoApi([root_dir '/annotations/instances_train2014.json']);
    fprintf('train_train (%d) images\n', length(dataset.imdb_train{1}.data.images));
    dataset.imdb_train{2} = CocoApi([root_dir '/annotations/instances_valminusminival2014.json']);
    fprintf('train_val (%d) images\n', length(dataset.imdb_train{2}.data.images));
    dataset.roidb_train = cell(size(dataset.imdb_train));
    
    dataset.imdb_test.coco = CocoApi([root_dir '/annotations/instances_minival2014.json']);
    fprintf('val_val (%d) images\n', length(dataset.imdb_test.coco.data.images));
    dataset.roidb_test = struct();
    conf.COCO.train_root{1} = [root_dir '/train2014'];
    conf.COCO.train_root{2} = [root_dir '/val2014'];
    conf.COCO.test_root = [root_dir '/val2014'];
    
elseif strcmp(conf.dataset, 'imagenet_3k')
    
    root_dir = './data/datasets/imagenet_3k';
    cprintf('blue', 'loading Extended ImageNet dataset ... \n');
    conf.EI.train_root = [root_dir '/cls_3k_train_im_v0'];
    conf.EI.test_root = [root_dir '/cls_3k_val_im_v0'];
    ld = load('data/datasets/imagenet_3k/image_3k_subset_v1.mat');
    conf.EI.train_list = ld.train_list;
    conf.EI.val_list = ld.val_list; clear ld;
    
    fprintf('train (%d) images\n', length(conf.EI.train_list));
    fprintf('val (%d) images\n', length(conf.EI.val_list));
    % useless, for legacy reason
    dataset.imdb_train{1} = []; 
    dataset.roidb_train = cell(size(dataset.imdb_train));
    
elseif strcmp(conf.dataset, 'voc')
    dataset.imdb_train{1} = []; 
    dataset.roidb_train = cell(size(dataset.imdb_train));
end



