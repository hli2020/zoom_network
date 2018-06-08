function scaled_rois = scale_rois(rois, im_size, im_size_resize)
% add the following to prevent empty box in roidb
if isempty(rois)
    scaled_rois=[];
    return
end
scale = (im_size_resize - 1) ./ (im_size - 1);
scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end