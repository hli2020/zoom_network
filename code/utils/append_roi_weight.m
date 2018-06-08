function caffe_solver = append_roi_weight(solver_def_file, gpu_id, init_net_set)

resNet_caffe = caffe.Solver(init_net_set{3}, 0);
resNet_caffe.use_caffemodel(init_net_set{2});

% save the weights of resNet
resNet_layer_names = resNet_caffe.nets{1}.layer_names;
resNet_weights = [];
cnt = 0;
for i = 1:length(resNet_layer_names)
    
    curr_layer_name = resNet_layer_names{i};
    for j = 1:10
        try
            curr_weight = resNet_caffe.nets{1}.params(curr_layer_name, j).get_data();
            if j == 1, cnt = cnt + 1; end
            resNet_weights(cnt).layer_name = curr_layer_name;
            resNet_weights(cnt).weights{j} = curr_weight;
        catch
            break;
        end
    end
    
end
caffe.reset_all;
stard_ind = 436;    % 'res5a_branch1'
end_ind = 465;      % 'scale5c_branch2'

caffe_solver = caffe.Solver(solver_def_file, gpu_id);
net_names_debug = caffe_solver.nets{1}.layer_names;

fprintf('copy first half weights from D15a model ...\n');
caffe_solver.use_caffemodel(init_net_set{1});

fprintf('copy second half weights from resNet model ...\n');
gpu_num = length(gpu_id);
for i = stard_ind : end_ind
    for kk = 1:3
        curr_layer_name = sprintf('RPN%d_%s', kk, resNet_weights(i).layer_name);
        not_copy = false;
        
        for j = 1:length(resNet_weights(i).weights)
            for mm = 1:gpu_num
                try
                    caffe_solver.nets{mm}.set_params_data(curr_layer_name, j, ...
                        resNet_weights(i).weights{j});
                catch
                    warning('do not copy (%s) layer \n', curr_layer_name);
                    not_copy = true;
                    break;
                end
            end
            if not_copy, break; end
        end
    end
end
