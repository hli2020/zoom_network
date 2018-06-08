% Object proposals using deep models
%   Author:             Hongyang Li
%   Affiliation:        Chinese Univ. of Hong Kong
%   Date:               Jan 17, 2017
%   Email:              yangli@ee.cuhk.edu.hk
%   Origin(model_id):   D17b_roi_coco

close all;
clear;
caffe.reset_all;

%% first things first: init
% will create 'train' and 'test' sub-folders  
conf.model_id                           = mfilename();
conf.dataset                            = 'voc';
conf.debug                              = true; %false;
conf.draw_boxes                         = false;
conf                                    = default_config('train', conf);

% || optional parameters
conf.gpu_id                             = 0;
conf.rpn_max_size                       = 800;
conf.save_interval                      = 5000;
conf.add_gray_cls                       = true; %false;
conf.multi_depth                        = true; %false;
conf.check_certain_scale                = true; %false; % hg structure
conf.fg_thresh                          = 0.51;
conf.batch_size                         = 300;
% anchor
conf.rpn_feat_stride                    = [8, 16, 32];
conf.base_size                          = 8;
conf.liuyu.range                        = [72 128];
conf.anchor_scale                       = 2.^(0:5);  % default
conf.ratios                             = [0.25, 0.5, 1, 2, 3];
conf.loss_cls_weight                    = [1 1 1];
conf.loss_bbox_weight                   = [1 1 1];
% init
conf.init_mirror_layer                  = false;

conf= init_zoom_net(conf);
% located in the 'model/zoom' folder
model.solver_file                       = 'bn/set_8_balance_3cls_voc/solver_train_auto';
model.init_net_file                     = conf.init_net_file;
model.solverstate                       = ''; %'iter_5';

%% train zoom network
conf = orderfields(conf);
cprintf('blue', '\nBegin - Phase 1 Zoom Network TRAINING ...\n');
zoom_train(...
    'conf',                     conf, ...
    'solverstate',              model.solverstate, ...
    'solver',                   model.solver_file , ...
    'init_net_file',            model.init_net_file);
cprintf('blue', '\nDone - Phase 1 Zoom Network TRAINING ...\n');
exit;
