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
% will create 'train' and 'test' sub-folders  
conf.model_id                           = 'E02_imagenet_3k';  
conf.debug                              = false;
conf.draw_boxes                         = false;
% make sure total_chunk is less or equal than the number of GPUs
conf.test.total_chunk                   = 4;
conf.test.curr_chunk                    = 1;
conf.gpu_id                             = conf.test.curr_chunk - 1;

% || default setting
[conf, dataset]                         = init_rpn('test', conf);
% located in the 'model/zoom' folder
model.solver_file                       = 'bn/set_8_balance_3cls_coco/solver_deploy';
model.image_mean                        = conf.image_mean;
% will find the model under 'output/[conf.model_id]/train/[model.iter_name].caffemodel'
model.iter_name                         = 'iter_50000';
% || optional parameters


%% train zoom network
conf = orderfields(conf);
cprintf('blue', '\nBegin - Phase 1 Zoom Network TEST ...\n');
model.recall = zoom_test(...
    'trained_model_folder',     conf.data.train_key, ...
    'iter_name',                conf.test.which_iter, ...
    'model',                    model, ...
    'imdb',                     dataset.imdb_test, ...
    'roidb',                    dataset.roidb_test, ...
    'test_res_folder_suffix',   conf.test.res_folder_suffix, ...
    'debug',                    conf.debug, ...
    'conf',                     conf);
cprintf('blue', '\nDone - Phase 1 Zoom Network TEST ...\n');
exit;
