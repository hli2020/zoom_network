function [curr_level_output, inter_output] = process_test_output(conf, curGPU_info)
% curr_level_output is of size [anchor_num x 5] on a given level and scale
inter_output            = [];
curr_level              = conf.temp_curr_level;
output_blob             = curGPU_info.output_blob;
ori_im_size             = curGPU_info.ori_im_size;
scaled_im_size          = curGPU_info.scaled_im_size;
blob_match_template     = @(x) (strcmp(extractfield(output_blob, 'blob_name'), x));

if conf.roi_followup
    
    % 'roi_bbox_pred1'
    box_deltas = output_blob(blob_match_template(sprintf('roi_bbox_pred%d', curr_level))).data(5:8, :)';
    % 'roi_cls_prob1'
    scores = output_blob(blob_match_template(sprintf('roi_cls_prob%d', curr_level))).data(2, :)';
    
    boxes = conf.temp_raw_rescale_rois;
    pred_boxes = fast_rcnn_bbox_transform_inv(boxes, box_deltas);
    % scale back
    pred_boxes = bsxfun(@times, pred_boxes - 1, ...
        ([ori_im_size(2), ori_im_size(1), ori_im_size(2), ori_im_size(1)] - 1) ./ ...
        ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    pred_boxes = clip_boxes(pred_boxes, ori_im_size(2), ori_im_size(1));
    curr_level_output = [pred_boxes scores];
    
else
    
    % 'proposal_bbox_pred1'
    box_deltas = output_blob(blob_match_template(sprintf('proposal_bbox_pred%d', curr_level))).data;
    % 'proposal_cls_prob1'
    scores = output_blob(blob_match_template(sprintf('proposal_cls_prob%d', curr_level))).data(:, :, 2);
    scores = reshape(scores, size(box_deltas, 1), size(box_deltas, 2), []);
    curr_level_output = further_process_output(...
        conf, box_deltas, scores, scaled_im_size, ori_im_size);
end


if length(output_blob) == 12
    % means there are intermediate supervision results
    box_deltas_mi = output_blob(blob_match_template(sprintf('proposal_bbox_pred%d_mi', curr_level))).data;
    % 'proposal_cls_prob1'
    scores_mi = output_blob(blob_match_template(sprintf('proposal_cls_prob%d_mi', curr_level))).data(:, :, 2);
    scores_mi = reshape(scores_mi, size(box_deltas_mi, 1), size(box_deltas_mi, 2), []);
    inter_output = further_process_output(...
        conf, box_deltas_mi, scores_mi, scaled_im_size, ori_im_size);
end

end

function output = further_process_output(conf, box_deltas, scores, scaled_im_size, ori_im_size)
%% box_deltas
featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
% permute from [width, height, channel] to [channel, height, width],
% where channel is the fastest dimension
box_deltas = permute(box_deltas, [3, 2, 1]);
box_deltas = reshape(box_deltas, 4, [])';
% disp(scaled_im_size);
% disp(featuremap_size);

anchors = proposal_locate_anchors(conf, scaled_im_size, featuremap_size);
pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
% scale back
pred_boxes = bsxfun(@times, pred_boxes - 1, ...
    ([ori_im_size(2), ori_im_size(1), ori_im_size(2), ori_im_size(1)] - 1) ./ ...
    ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
pred_boxes = clip_boxes(pred_boxes, ori_im_size(2), ori_im_size(1));

%% scores
scores = permute(scores, [3, 2, 1]);
scores = scores(:);

% drop too small boxes
[pred_boxes, scores] = filter_boxes(conf.test.min_box_size, pred_boxes, scores);
[scores, scores_ind] = sort(scores, 'descend');
pred_boxes = pred_boxes(scores_ind, :);
% size: anchor_num x 5
output = [pred_boxes, scores];

end

function [boxes, scores] = filter_boxes(min_box_size, boxes, scores)
widths = boxes(:, 3) - boxes(:, 1) + 1;
heights = boxes(:, 4) - boxes(:, 2) + 1;

valid_ind = widths >= min_box_size & heights >= min_box_size;
boxes = boxes(valid_ind, :);
scores = scores(valid_ind, :);
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