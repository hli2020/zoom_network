function im_resize = prep_im_for_blob(im, target_size, conf)
% target_size is the shorter_dim
% this function is both used for training and test

im_means  = conf.DATA.image_mean;
if strcmp(conf.mode, 'train')
    max_size = conf.rpn_max_size;
else
    max_size = conf.test_multiscale_max_size;
end
wh = [size(im,1) size(im,2)];
shorter = min(wh); longer = max(wh);
resize_factor = target_size/shorter;

potential_max_size = (target_size/shorter)*longer;
if potential_max_size > max_size
    resize_factor = max_size/longer;
end
new_h = floor(resize_factor*size(im,1));
new_w = floor(resize_factor*size(im,2));

if conf.check_certain_scale    
    % find the nearest scale
    scale_set = conf.certain_scale;
    new_h = switch_to_nearest_scale(new_h, scale_set);
    new_w = switch_to_nearest_scale(new_w, scale_set);
end

im_resize = imresize(im, [new_h new_w], 'bilinear', 'antialiasing', false);
im_resize = single(im_resize);
try
    im_resize = bsxfun(@minus, im_resize, im_means);
catch
    im_means = imresize(im_means, [size(im_resize, 1), size(im_resize, 2)], ...
        'bilinear', 'antialiasing', false);
    im_resize = bsxfun(@minus, im_resize, im_means);
end
end

function new = switch_to_nearest_scale(old, scale_set)
[~, ind] = min((scale_set - old).^2);
new = scale_set(ind);
end
