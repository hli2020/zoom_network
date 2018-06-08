function new_scale = dynamic_change_scale(conf, gt_rois, im_size)

% ugly_size_set = conf.base_size .* conf.anchor_scale;
% % randomly choose one anchor size and one GT
% ugly_size = ugly_size_set(randi(length(ugly_size_set)));

% liuyu:
ugly_size = randi(conf.liuyu.range);

chosen_gt = gt_rois(randi(size(gt_rois, 1)), :);
gt_shorter_dim = min(chosen_gt(3)-chosen_gt(1), chosen_gt(4)-chosen_gt(2));
im_shorter_dim = min(im_size(1), im_size(2));

new_scale = round( ( ugly_size / gt_shorter_dim ) * im_shorter_dim );
end
