function aboxes = boxes_filter_inline_normal(aboxes, ...
    per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)

% to speed up nms (current = -1)
% nms_overlap_thres = 0.7
% after_nms_topN = 2000
if per_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
end
% do nms
show_num = 1000;
% fprintf('do nms during test, taking quite a while (brew some coffe or take a walk!:)...\n');
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    if use_gpu
        for i = 1:length(aboxes)
%             if i == 1 || i == length(aboxes) || mod(i, show_num)==0
%                 fprintf('\tgpu nms \t\t(%d/%d) \n', i, length(aboxes));
%             end
            aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
        end
    else
        parfor i = 1:length(aboxes)
            aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
        end
    end
end
aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);

if after_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(size(x,1), after_nms_topN), :), aboxes, 'UniformOutput', false);
end
end


