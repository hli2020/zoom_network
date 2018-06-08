function to_show_proposal = process_regression_result(to_show_proposal, raw_boxes_roi, choice)

if nargin <= 2
    choice = 'naive_1';
end

switch choice
    case 'naive_1'
        % average all boxes refinement across levels and scales, scores
        % won't (aka, ranking) change
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i}(:, 1:4);
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        to_show_proposal{end+1} = mean(out, 3);
        
    case 'naive_1_nms5'
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i}(:, 1:4);
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        nms_out = boxes_filter_inline([mean(out, 3) origin_score], ...
            -1, 0.5, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
    case 'naive_1_nms6'
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i}(:, 1:4);
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        nms_out = boxes_filter_inline([mean(out, 3) origin_score], ...
            -1, 0.6, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
    case 'naive_1_nms7'
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i}(:, 1:4);
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        nms_out = boxes_filter_inline([mean(out, 3) origin_score], ...
            -1, 0.7, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
    case 'naive_2'
        
        % average all results and refine its score by roi_score + origin
        % score, and rerank the boxes
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        out_almost(:, end) = out_almost(:, end) + origin_score;
        [~, ind] = sort(out_almost(:, end), 'descend');
        to_show_proposal{end+1} = out_almost(ind, :);
        
    case 'final_nms6'
        
        % abandon original scores and use refined boxes and scores together
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        nms_out = boxes_filter_inline(out_almost, -1, 0.6, 2000, true);
        to_show_proposal{end+1} = nms_out;
                
    case 'final_nms7'
        
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        nms_out = boxes_filter_inline(out_almost, -1, 0.7, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
        %     case 'naive_3'
        %
        %         % not tried
        %         percent = 0.7;
        %         origin_score = raw_boxes_roi{1, 1}(:, 5);
        %         origin_boxes = raw_boxes_roi{1, 1}(:, 1:4);
        %
        %         raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        %         out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        %         for i = 1:size(raw_boxes_roi, 2)     % each scale
        %
        %             curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
        %             for j = 1:size(raw_boxes_roi, 1)        % each level
        %                 curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
        %             end
        %             out(:, :, i) = mean(curr_scale_box, 3);
        %         end
        %
        %         out_almost = mean(out, 3);
        %         change = (out_almost(:, end) - origin_score).^2;
        %         [value, ind] = sort(change, 'ascend');
        %         need_to_change_id
        %         to_show_proposal{end+1} = out_almost(ind, :);
        
    case 'naive_2_nms5'
        
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        out_almost(:, end) = out_almost(:, end) + origin_score;
        [~, ind] = sort(out_almost(:, end), 'descend');
        
        nms_out = boxes_filter_inline(out_almost(ind, :), ...
            -1, 0.5, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
    case 'naive_2_nms6'
        
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        out_almost(:, end) = out_almost(:, end) + origin_score;
        [~, ind] = sort(out_almost(:, end), 'descend');
        
        nms_out = boxes_filter_inline(out_almost(ind, :), ...
            -1, 0.6, 2000, true);
        to_show_proposal{end+1} = nms_out;
              
    case 'naive_2_nms7'
        
        origin_score = raw_boxes_roi{1, 1}(:, 5);
        raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        out = zeros(size(raw_boxes_roi{1, 1}, 1), 5, size(raw_boxes_roi, 2));
        for i = 1:size(raw_boxes_roi, 2)     % each scale
            
            curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 5, size(raw_boxes_roi, 1));
            for j = 1:size(raw_boxes_roi, 1)        % each level
                curr_scale_box(:, :, j) = raw_boxes_roi{j, i};
            end
            out(:, :, i) = mean(curr_scale_box, 3);
        end
        
        out_almost = mean(out, 3);
        out_almost(:, end) = out_almost(:, end) + origin_score;
        [~, ind] = sort(out_almost(:, end), 'descend');
        
        nms_out = boxes_filter_inline(out_almost(ind, :), ...
            -1, 0.7, 2000, true);
        to_show_proposal{end+1} = nms_out;
        
        %     case 'naive_3a'
        %         raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        %         out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        %         for i = 1:size(raw_boxes_roi, 2)     % each scale
        %
        %             curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, 1);
        %             for j = 1 % each level
        %                 curr_scale_box(:, :, j) = raw_boxes_roi{j, i}(:, 1:4);
        %             end
        %             out(:, :, i) = mean(curr_scale_box, 3);
        %         end
        %         to_show_proposal{end+1} = mean(out, 3);
        %
        %     case 'naive_3b'
        %         raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        %         out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        %         for i = 1:size(raw_boxes_roi, 2)     % each scale
        %
        %             curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, 1);
        %             for j = 2 % each level
        %                 curr_scale_box(:, :, 1) = raw_boxes_roi{j, i}(:, 1:4);
        %             end
        %             out(:, :, i) = mean(curr_scale_box, 3);
        %         end
        %         to_show_proposal{end+1} = mean(out, 3);
        %
        %     case 'naive_3c'
        %         raw_boxes_roi = raw_boxes_roi(2:end, :); % remove original results
        %         out = zeros(size(raw_boxes_roi{1, 1}, 1), 4, size(raw_boxes_roi, 2));
        %         for i = 1:size(raw_boxes_roi, 2)     % each scale
        %
        %             curr_scale_box = zeros(size(raw_boxes_roi{1, i}, 1), 4, 1);
        %             for j = 3 % each level
        %                 curr_scale_box(:, :, 1) = raw_boxes_roi{j, i}(:, 1:4);
        %             end
        %             out(:, :, i) = mean(curr_scale_box, 3);
        %         end
        %         to_show_proposal{end+1} = mean(out, 3);
end

end