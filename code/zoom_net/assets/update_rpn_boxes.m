function recursive_rpn_box = update_rpn_boxes(...
    caffe_solver, level_num, net_inputs, all_rois_blob, rois_per_iter, image_size)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
caffe_solver.set_phase('test');
reg_channel_num = size(net_inputs{1}{4}, 3);
gpu_num = length(caffe_solver.nets);
layer_name_template = @(x) sprintf('roi_bbox_pred%d', x);
recursive_rpn_box = cell(gpu_num, 1);

%%
% prepare net_inputs, note that each input may have different roi_num
roi_num_longest = 0;
for i = 1 : gpu_num
    
    roi_num = size(all_rois_blob{i}, 4);
    input_block = {all_rois_blob{i}, zeros(1, 1, 1, roi_num), ...
        zeros(1, 1, reg_channel_num, roi_num), zeros(1, 1, reg_channel_num, roi_num)};
    
    for kk = 1:level_num
        start_ind = 2+(kk-1)*4;
        end_ind = 5+(kk-1)*4;
        net_inputs{i}(start_ind:end_ind) = input_block;
    end
    if roi_num >= roi_num_longest, roi_num_longest = roi_num; end
end

% split into smaller chunks
for kk = 1 : ceil(roi_num_longest / rois_per_iter)
    
    sub_ind_start = 1 + (kk-1) * rois_per_iter;
    sub_ind_end = min(roi_num_longest, kk * rois_per_iter);
    
    sub_net_inputs = cell(gpu_num, 1);
    for i = 1 : gpu_num
        
        sub_net_inputs{i}{1} = net_inputs{i}{1}; % image blob
        curr_im_roi_num = size(net_inputs{i}{2}, 4);
        
        for j = 2:length(net_inputs{i})
            
            temp = net_inputs{i}{j}; % the j-th blob (label_blob, roi_blob, etc) in i-th image    
            if sub_ind_end <= curr_im_roi_num
                sub_net_inputs{i}{j} = temp(:, :, :, sub_ind_start:sub_ind_end);
            else
                % say 761:800, yet temp (gpu2) only has 777 ROIs
                % fill the rest with zeros
                total_sub_length = sub_ind_end - sub_ind_start + 1;
                channel_num = size(temp, 3);
                
                if curr_im_roi_num >= sub_ind_start
                    sub_net_inputs{i}{j} = temp(:, :, :, sub_ind_start:curr_im_roi_num);
                    sub_net_inputs{i}{j}(:, :, channel_num, end+1: total_sub_length) = zeros;
                else
                    
                    sub_net_inputs{i}{j}(:, :, channel_num, 1:total_sub_length) = zeros;
                end
            end
        end
    end
    
    caffe_solver.reshape_as_input(sub_net_inputs);
    caffe_solver.forward(sub_net_inputs);

    for i = 1 : gpu_num
        
        % process output in each gpu/image
        im_size = image_size{i}.origin;
        scaled_im_size = image_size{i}.resize;
        ori_boxes = squeeze(all_rois_blob{i})'; % [1675 x 4]
        ori_boxes(end+1:roi_num_longest, :) = zeros;
        
        boxes = ori_boxes(sub_ind_start:sub_ind_end, 2:end) + 1;
        % store results for three levels
        pred_boxes = zeros(size(boxes, 1), size(boxes, 2), level_num);
        
        for j = 1:level_num
            
            output = ...
                caffe_solver.nets{i}.blobs(layer_name_template(j)).get_data();
            reg_output = output(5:8, :)';
            curr_level_pred_boxes = fast_rcnn_bbox_transform_inv(boxes, reg_output);
            % scale back
            curr_level_pred_boxes = bsxfun(@times, curr_level_pred_boxes - 1, ...
                ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ...
                ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
            pred_boxes(:, :, j) = clip_boxes(curr_level_pred_boxes, im_size(2), im_size(1));           
        end
        
        recursive_rpn_box{i}(sub_ind_start:sub_ind_end, :) = mean(pred_boxes, 3);
    end
end  % loopr for each chunk (the longest)

for i = 1 : gpu_num
    curr_im_roi_num = size(net_inputs{i}{2}, 4);
    recursive_rpn_box{i} = recursive_rpn_box{i}(1:curr_im_roi_num, :);
end
caffe_solver.set_phase('train');
end

function boxes = clip_boxes(boxes, im_width, im_height)
% x1 >= 1 & <= im_width
boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
% y1 >= 1 & <= im_height
boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
% x2 >= 1 & <= im_width
boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
% y2 >= 1 & <= im_height
boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end