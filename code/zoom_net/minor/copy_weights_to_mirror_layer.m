function caffe_solver = copy_weights_to_mirror_layer(caffe_solver, gpu_num)

layer_name = caffe_solver.nets{1}.layer_names;
for i = 1:length(layer_name)
    
    copyed_ = false;
    curr_layer_name = layer_name{i};
    if length(curr_layer_name) >= 10 && strcmp(curr_layer_name(1:10), 'inception_')
        mirror_name = [curr_layer_name(1:12) '_mirror' curr_layer_name(13:end)];
        
        if find(strcmp(mirror_name, layer_name))
            for j = 1:10
                try
                    curr_weight = caffe_solver.nets{1}.params(curr_layer_name, j).get_data();
                    for mm = 1:gpu_num
                            caffe_solver.nets{mm}.set_params_data(mirror_name, j, curr_weight);
  
                    end
                    copyed_ = true;
                catch
                    break;
                end
            end
            if copyed_
                fprintf('copy weights from (%s) to (%s) ...\n', curr_layer_name, mirror_name);
            end
        end
    end
end