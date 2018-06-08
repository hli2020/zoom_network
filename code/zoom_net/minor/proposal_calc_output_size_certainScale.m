function [output_width_map, output_height_map, certain_scale] = ...
    proposal_calc_output_size_certainScale(...
    conf, solver_def_file, gpu_id, end_level)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

arch = 'bn_balance';
% this is the most customized part
check_list_blob_name{1}{1} = 'inception_3b/output';
check_list_blob_name{1}{2} = 'inception_3c_mirror_upsample';
check_list_blob_name{2}{1} = 'inception_4d/output';
check_list_blob_name{2}{2} = 'inception_4e_mirror_upsample';
check_list_blob_name{3}{1} = 'inception_5b/output';

if strcmp(arch, 'bn')
    
    check_list_blob_name{3}{2} = 'inception_d2_parallel/output';
    check_list_blob_name{3}{3} = 'inception_d3_parallel/output';
    
elseif strcmp(arch, 'bn_balance') || strcmp(arch, 'bn_roi_followUp')
    
    check_list_blob_name{3}{2} = 'inception_d2_para/output';
    
else
    error('unknown network structure');
end

certain_scale = [];
% caffe_log_file_base = fullfile('debug_');
% caffe.init_log(caffe_log_file_base);
% solver_def_file = [solver_def_file(1:end-9) '_check.prototxt'];
% init caffe net using Solver
caffe_solver = caffe.Solver(solver_def_file, gpu_id);
caffe_solver.set_phase('test');

output_w = cell(end_level, 1); output_h = cell(end_level, 1);
input = conf.min_size : conf.max_size;
for kk = 1:end_level, output_w{kk} = nan(size(input)); output_h{kk} = nan(size(input)); end
% for the purpose of saving memory
fix_size = 128;

fprintf('\nchecking range: %d - %d ...\n', input(1), input(end));

for i = 1:length(input)
    
    curr_size = input(i);
    im_blob = single(zeros(curr_size, fix_size, 3, 1));
    net_inputs = {{im_blob}};
    
    % Reshape net's input blobs
    caffe_solver.reshape_as_input(net_inputs);
    caffe_solver.forward(net_inputs);
    
    USE_CURR_SIZE = true;
    for kk = 1:end_level
        check_var = [];
        for check_blob_id = 1:length(check_list_blob_name{kk})
            output_name = check_list_blob_name{kk}{check_blob_id};
            spatial_size = caffe_solver.nets{1}.blobs(output_name).get_data();
            check_var = [check_var size(spatial_size, 1)];
        end
        if sum(gradient(check_var)) ~= 0
            USE_CURR_SIZE = false;
            break;
        else
            output_w{kk}(i) = check_var(1);
            output_h{kk}(i) = check_var(1);
        end
    end
    
    if ~USE_CURR_SIZE
        % fprintf('\tsize %d is skipped ...\n', curr_size);
    else
        certain_scale = [certain_scale curr_size];
        fprintf('\tsize %d is saved ...\n', curr_size);
    end
end

for kk = 1:end_level
    output_width_map{kk} = containers.Map(input, output_w{kk});
    output_height_map{kk} = containers.Map(input, output_h{kk});
end
caffe.reset_all();
end
