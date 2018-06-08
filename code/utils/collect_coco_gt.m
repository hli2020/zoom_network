function [roidb_merge, empty_gt] = collect_coco_gt(cocoApi, input)
roidb_merge = [];
empty_gt = false;

% input: im_id, or im_name (without extension)
% cocoApi can be cell of many Apis, or a single structure

if isnumeric(input)
    % since we only got two different APIs
    try
        annoIds = cocoApi{1}.getAnnIds('imgIds', input);
        anno = cocoApi{1}.loadAnns(annoIds);
    catch
        annoIds = cocoApi{2}.getAnnIds('imgIds', input);
        anno = cocoApi{2}.loadAnns(annoIds);
    end
else
    list = cocoApi.data.images;
    im_id = list(strcmp(extractfield(list, 'file_name'), [input '.jpg'])).id;
    annoIds = cocoApi.getAnnIds('imgIds', im_id);
    anno = cocoApi.loadAnns(annoIds);
end

if isempty(anno)
    empty_gt = true;
    return;
end

roidb_merge.class = extractfield(anno, 'category_id')';
boxes = reshape(extractfield(anno, 'bbox'), 4, [])';
roidb_merge.boxes = [boxes(:, 1) boxes(:, 2) ...
    (boxes(:, 1)+boxes(:, 3)) (boxes(:, 2)+boxes(:, 4))];
end