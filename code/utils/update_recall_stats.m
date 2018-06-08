function [recall, gt_resize_log] = update_recall_stats(...
    recall, gt_resize_log, gt_info, gt_stats)

cls_list = unique(gt_info(:,1));
level_num = 3;

im_resize = gt_stats.im_size_resize(1:2);
for i = 1:length(cls_list)

    curr_cls = cls_list(i);
    detect_mat = gt_info(gt_info(:, 1)==curr_cls, 2:end);
    curr_cls_total_num = sum(gt_info(:, 1)==curr_cls);
    curr_cls_all_level_vec = zeros(curr_cls_total_num, 1);
    for kk = 1:level_num
        curr_cls_curr_level_correct_num = sum(detect_mat(:, kk));
        curr_cls_all_level_vec = curr_cls_all_level_vec | detect_mat(:, kk);
        recall.(sprintf('level_%d', kk))(curr_cls, 1) = ...
            curr_cls_total_num + recall.(sprintf('level_%d', kk))(curr_cls, 1);
        recall.(sprintf('level_%d', kk))(curr_cls, 2) = ...
            curr_cls_curr_level_correct_num + recall.(sprintf('level_%d', kk))(curr_cls, 2);
    end
    recall.all(curr_cls, 1) = curr_cls_total_num + recall.all(curr_cls, 1);
    recall.all(curr_cls, 2) = sum(curr_cls_all_level_vec) + recall.all(curr_cls, 2);
    
    gt_resize_log{curr_cls, 1} = cat(1, ...
        gt_resize_log{curr_cls, 1}, ...
        gt_stats.gt_resize_rois(gt_info(:, 1)==curr_cls, 2:end));
    
    gt_resize_log{curr_cls, 2} = cat(1, ...
        gt_resize_log{curr_cls, 2}, ...
        repmat(im_resize, [curr_cls_total_num 1]));
end
end