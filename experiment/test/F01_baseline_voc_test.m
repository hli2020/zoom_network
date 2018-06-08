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
conf.dataset                            = 'voc';
% will create 'train' and 'test' sub-folders  
conf.model_id                           = mfilename();
conf.debug                              = false;
conf.draw_boxes                         = false;
conf                                    = default_config('test', conf);

conf.total_chunk                        = 2;
conf.curr_chunk                         = 2;
% || optional parameters
conf.gpu_id                             = 1;
conf.add_gray_cls                       = false;
conf.multi_depth                        = false;
conf.check_certain_scale                = false;
conf.fg_thresh                          = 0.51;
% anchor
conf.rpn_feat_stride                    = 32;
conf.base_size                          = 8;
conf.liuyu.range                        = [72 128];
conf.anchor_scale                       = [8 16];
conf.ratios                             = [0.5, 1, 2];

conf= init_zoom_net(conf);
% located in the 'model/zoom' folder
root = pwd;
model.solver_file = 'bn/baseline/solver_deploy';
model.test_caffemodel = fullfile(root, 'output', ...
    'F01_baseline_voc_drop_lr2', 'train', 'iter_50000.caffemodel');

%% test zoom network
conf = orderfields(conf);
cprintf('blue', '\nBegin - Phase 1 Zoom Network TESTING ...\n');
zoom_test(...
    'conf',                     conf, ...
    'solver',                   model.solver_file , ...
    'test_model',               model.test_caffemodel);
cprintf('blue', '\nDone - Phase 1 Zoom Network TESTING ...\n');
exit;
