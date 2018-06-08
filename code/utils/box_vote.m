function output = box_vote(input_pool, nms_choose_ind, vote_thres)

output = zeros(length(nms_choose_ind), 5);
nms_choose_box = input_pool(nms_choose_ind, 1:4);
[true_overlap, ~] = compute_overlap_hyli(nms_choose_box, input_pool(:, 1:4));

for i = 1:length(true_overlap)
    to_merge_box = input_pool(true_overlap(i).overlap >= vote_thres, :);
    output(i, :) = mean(to_merge_box, 1);
end
end
