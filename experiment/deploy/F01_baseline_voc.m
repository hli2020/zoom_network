% Object proposals using deep models
%   Author:             Hongyang Li
%   Affiliation:        Chinese Univ. of Hong Kong
%   Date:               Jan 17, 2017
%   Email:              yangli@ee.cuhk.edu.hk
%   Origin(model_id):   D17b_roi_coco


% adjust the model mannually during training
close all;
clear;
caffe.reset_all;

%% configure: dataset, model, params
% || essential setup in each experiment, for other parameters, see 'init_rpn.m'
% dataset options: coco, imagenet, pascal
conf.dataset                            = 'voc';
% will create 'train' and 'test' sub-folders  
conf.model_id                           = mfilename();
manual_train_suffix                     = '_drop_lr2';
trained_from_suffix                     = '_drop_lr1';
conf.model_id                           = [mfilename() manual_train_suffix];
conf.debug                              = false;
conf.draw_boxes                         = false;
% || default setting
[conf, dataset]                         = default_config('train', conf);

% || optional parameters
conf.gpu_id                             = 0; %3;
conf.rpn_max_size                       = 1000;
conf.save_interval                      = 5000;
conf.add_gray_cls                       = false;
conf.multi_depth                        = false;
conf.check_certain_scale                = false;
conf.fg_thresh                          = 0.51;
conf.batch_size                         = 300;
% anchor
conf.rpn_feat_stride                    = 32;
conf.base_size                          = 8;
conf.liuyu.range                        = [72 128];
conf.anchor_scale                       = [8 16];
conf.ratios                             = [0.5, 1, 2];
conf.loss_cls_weight                    = 1;
conf.loss_bbox_weight                   = 1;

conf= init_zoom_net(conf);
% located in the 'model/zoom' folder
model.solver_file                       = 'bn/baseline/solver_train';
% model.init_net_file                     = conf.init_net_file;
model.init_net_file = fullfile(pwd, 'output', ...
    [mfilename() trained_from_suffix], 'train', ...
    'iter_50000.caffemodel');

%% train zoom network
conf = orderfields(conf);
cprintf('blue', '\nBegin - Phase 1 Zoom Network TRAINING ...\n');
zoom_train(...
    'conf',                     conf, ...
    'solver',                   model.solver_file , ...
    'init_net_file',            model.init_net_file);
cprintf('blue', '\nDone - Phase 1 Zoom Network TRAINING ...\n');
exit;
