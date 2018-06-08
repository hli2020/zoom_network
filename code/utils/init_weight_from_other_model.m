function caffe_solver = init_weight_from_other_model(...
    weight_detail, pretrain_model, solver_def_file, gpu_id)
% IMPORTANT note:
%       we use uniformly 'solver_def_file' as the network structure, which
%       is the same as the to-be-trained model/task; in some cases, the ft
%       model from RPN is shorter (say 4b32), the 'ft_weights' still has
%       data in higher layers (say 4b35), but thosee params are initiated
%       as DEFAULT or as said in the solver file.

% jot down the weights and clear solvers
solver_pretrain = caffe.Solver(solver_def_file, 0);
solver_pretrain.use_caffemodel(pretrain_model);
pretrain_layer_names = solver_pretrain.nets{1}.layer_names;
pretrain_weights = [];
cnt = 0;
for i = 1:length(pretrain_layer_names)
    
    curr_layer_name = pretrain_layer_names{i};
    for j = 1:10
        try
            curr_weight = solver_pretrain.nets{1}.params(curr_layer_name, j).get_data();
            if j == 1, cnt = cnt + 1; end
            pretrain_weights(cnt).layer_name = curr_layer_name;
            pretrain_weights(cnt).weights{j} = curr_weight;
        catch
            break;
        end
    end
    
end
caffe.reset_all;

solver_rpn_finetune = caffe.Solver(solver_def_file, 0);
solver_rpn_finetune.use_caffemodel(weight_detail(1).init_from);
ft_layer_names = solver_rpn_finetune.nets{1}.layer_names;
ft_weights = [];
cnt = 0;
for i = 1:length(ft_layer_names)
    
    curr_layer_name = ft_layer_names{i};
    for j = 1:10
        try
            curr_weight = solver_rpn_finetune.nets{1}.params(curr_layer_name, j).get_data();
            if j == 1, cnt = cnt + 1; end
            ft_weights(cnt).layer_name = curr_layer_name;
            ft_weights(cnt).weights{j} = curr_weight;
        catch
            break;
        end
    end
    
end
caffe.reset_all;

% assign the weights
caffe_solver = caffe.Solver(solver_def_file, gpu_id);
num_gpu = length(gpu_id);
end_layer_id = find( strcmp(extractfield(ft_weights, 'layer_name'), ...
    weight_detail(1).end_layer) == 1);

% first from ft model
cprintf('blue', 'load (ft RPN model) params from layer "%s" to "%s"\n\n', ...
    ft_weights(1).layer_name, ft_weights(end_layer_id).layer_name);

for i = 1 : end_layer_id
    curr_layer_name = ft_weights(i).layer_name;
    for j = 1:length(ft_weights(i).weights)
        curr_weights = ft_weights(i).weights{j};
        for kk = 1:num_gpu
            caffe_solver.nets{kk}.set_params_data(curr_layer_name, j, curr_weights);
        end
    end
end

% then from pretrained model for the rest layers
cprintf('blue', 'load (pretrained model) params from layer "%s" to "%s"\n\n', ...
    ft_weights(end_layer_id+1).layer_name, ft_weights(end).layer_name);

for i = (end_layer_id+1) : length(pretrain_weights)
    curr_layer_name = pretrain_weights(i).layer_name;
    for j = 1:length(pretrain_weights(i).weights)
        curr_weights = pretrain_weights(i).weights{j};
        for kk = 1:num_gpu
            caffe_solver.nets{kk}.set_params_data(curr_layer_name, j, curr_weights);
        end
    end
end

fprintf('network weight init done (from different models)!\n');
