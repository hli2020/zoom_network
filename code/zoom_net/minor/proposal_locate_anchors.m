function anchors = proposal_locate_anchors(...
    conf, im_size_resize, feature_map_size)
% refactored by Hongyang, generate anchors for each scale PER image
% update:   'im_size' is the resized image
%
% train:    'proposal_generate_minibatch_on_the_fly.m'
% test:     'proposal_im_detect.m'
% for train, the input argument is the first two, and for test, all three
% input arguments are provided.

if ~exist('feature_map_size', 'var')
    feature_map_size = [];
end

anchors = proposal_locate_anchors_single_scale(...
    im_size_resize, conf, feature_map_size);

end

function anchors = proposal_locate_anchors_single_scale(...
    im_size_resize, conf, feature_map_size)

curr_level = conf.temp_curr_level;

if isempty(feature_map_size)
    
    output_size = cell2mat([conf.output_height_map{curr_level}.values({im_size_resize(1)}), ...
        conf.output_width_map{curr_level}.values({im_size_resize(2)})]);
else
    output_size = feature_map_size;
end

shift_x = (0:(output_size(2)-1)) * conf.rpn_feat_stride(curr_level);
shift_y = (0:(output_size(1)-1)) * conf.rpn_feat_stride(curr_level);
[shift_x, shift_y] = meshgrid(shift_x, shift_y);

% concat anchors as [channel, height, width], where channel is the fastest dimension.
anchors = reshape(bsxfun(@plus, permute(conf.anchors{curr_level}, [1, 3, 2]), ...
    permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);

% equals to
% anchors = arrayfun(@(x, y) single(bsxfun(@plus, conf.anchors, [x, y, x, y])), ...
%     shift_x, shift_y, 'UniformOutput', false);
% anchors = reshape(anchors, [], 1);
% anchors = cat(1, anchors{:});

end