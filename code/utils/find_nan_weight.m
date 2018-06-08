function find_nan_weight(caffe_solver, iter)

layer_names = caffe_solver.nets{1}.layer_names;

for i = 1:length(layer_names)
    curr_layer_name = layer_names{i};
    for j = 1:10
        try
            curr_weight = caffe_solver.nets{1}.params(curr_layer_name, j).get_data();
            if any(isnan(curr_weight))
                fprintf('abs iter:: %d\tlayer\t%s (param id j=%d) has NaN\n', iter, curr_layer_name, j);
            end
        catch
            break;
        end
    end
end
end