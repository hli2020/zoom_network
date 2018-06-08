function imdb_info = get_imagenet_3k_info(input)

imdb_info.label = [];               % [100 x 1] double
imdb_info.imdb_merge_size = [];     % [100 x 2] double
imdb_info.imdb_merge_path = [];     % [100 x 1] cell
imdb_info.im_path_root = [];        % [1 x 1] cell

imdb_info.label = extractfield(input.train_list, 'label')';
bbox = struct2cell(input.train_list);
imdb_info.bbox = squeeze(bbox(2, :, :));

im_size = extractfield(input.train_list, 'size');
im_size = reshape(im_size, [2 length(input.train_list)])';
imdb_info.imdb_merge_size = double(im_size);

imdb_info.imdb_merge_path = extractfield(input.train_list, 'im_name')';
imdb_info.im_path_root{1} = input.train_root;
end