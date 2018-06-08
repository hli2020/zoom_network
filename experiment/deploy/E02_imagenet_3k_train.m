% Object proposals using deep models
%   Author:             Hongyang Li
%   Affiliation:        Chinese Univ. of Hong Kong
%   Date:               Jan 17, 2017
%   Email:              yangli@ee.cuhk.edu.hk
%   Origin(model_id):   D17b_roi_coco

close all;
clear;
caffe.reset_all;

%% configure: dataset, model, params
% || essential setup in each experiment, for other parameters, see 'init_rpn.m'
% dataset options: coco, imagenet, pascal
conf.dataset                            = 'imagenet_3k';
% conf.dataset                            = 'coco';
% will create 'train' and 'test' sub-folders  
% conf.model_id                           = 'E02_imagenet_3k';
conf.model_id                           = 'E02_imagenet_3k_resume';
conf.debug                              = false;
conf.draw_boxes                         = false;
% || default setting
[conf, dataset]                         = init_rpn('train', conf);
% located in the 'model/zoom' folder
% model.solver_file                       = 'bn/set_8_balance_3cls_coco/solver_train';
model.solver_file                       = 'bn/set_8_balance_3cls_coco/solver_train_resume';
model.init_net_file                     = conf.init_net_file;

% || optional parameters
conf.gpu_id                             = 0:3;
conf.rpn_max_size                       = 1000;
conf.save_interval                      = 5000;
conf.loss_bbox_weight                   = [10 10 5];
conf.init_net_file                      = fullfile(pwd, ...
    'output', 'E02_imagenet_3k_train', 'train', 'iter_50000.caffemodel');
conf.rng_seed                           = 2; % in resume we generate a different series of samples

%% train zoom network
conf = orderfields(conf);
cprintf('blue', '\nBegin - Phase 1 Zoom Network TRAINING ...\n');
model.trained_caffemodel = zoom_train(...
    'conf',                     conf, ...
    'imdb_train',               dataset.imdb_train, ...
    'roidb_train',              dataset.roidb_train, ...
    'solver',                   model.solver_file , ...
    'init_net_file',            model.init_net_file);
cprintf('blue', '\nDone - Phase 1 Zoom Network TRAINING ...\n');
exit;
