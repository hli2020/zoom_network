function [output_width_map, output_height_map] = ...
    proposal_calc_output_size(conf, test_solver_def_file, gpu_id, ...
    output_name, curr_level)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

% init caffe net using Solver
caffe_solver = caffe.Solver(test_solver_def_file, gpu_id);
caffe_solver.set_phase('test'); 

input = conf.min_size : conf.max_size;
output_w = nan(size(input));
output_h = nan(size(input));
fix_size = 100;

for i = 1:length(input)
    
    s = input(i);
    im_blob = single(zeros(s, fix_size, 3, 1));
    net_inputs = {{im_blob}};
    
    % Reshape net's input blobs
    caffe_solver.reshape_as_input(net_inputs);
    caffe_solver.forward(net_inputs);
    
    cls_score = caffe_solver.nets{1}.blobs(output_name).get_data();
    output_w(i) = size(cls_score, 1);
    output_h(i) = output_w(i);
    tic_toc_print('level m=%d, %d/%d,\t\tinput_hw, (%d(fake) x %d),\t\toutput_hw,(%d x %d)\n', ...
        curr_level, i, length(input), fix_size, s, output_h(i), output_w(i));
end

output_width_map = containers.Map(input, output_w);
output_height_map = containers.Map(input, output_h);

caffe.reset_all();
end
