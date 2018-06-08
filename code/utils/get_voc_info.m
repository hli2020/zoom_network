function imdb_info = get_voc_info(mode)

root1            = 'data/datasets/pascal/VOCdevkit/VOC2007';
im_path1         = fullfile(pwd, root1, 'JPEGImages');
root2            = 'data/datasets/pascal/VOCdevkit/VOC2012';
im_path2         = fullfile(pwd, root2, 'JPEGImages');

if strcmp(mode, 'train')
    
    ld = load('data/datasets/pascal/voc0712_trainval_gt_info.mat');
    % conf.VOC.im_path_root{1} = im_path1;
    % conf.VOC.im_path_root{2} = im_path2;
    imdb_info.im_path_root{1} = im_path1;
    imdb_info.im_path_root{2} = im_path2;
    
elseif strcmp(mode, 'test')
    
    ld = load('data/datasets/pascal/voc07_test_gt_info.mat');
    imdb_info.im_path_root{1} = im_path1;
end

imdb_info.imdb_merge_size = ld.imdb_merge_size;     % [100 x 2] double
imdb_info.imdb_merge_path = ld.imdb_merge_path;     % [100 x 1] cell
imdb_info.roidb_merge = ld.roidb_merge;
imdb_info.dataset = 'voc'; % for latter evaluation use
end