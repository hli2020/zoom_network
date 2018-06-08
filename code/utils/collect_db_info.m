function imdb_info = collect_db_info(imdbs, roidbs)
roidb_merge = []; imdb_merge_path = []; imdb_merge_size = [];
for i = 1:length(imdbs)
    assert( strcmp(imdbs{i}.name, roidbs{i}.name) );        % name match
    assert(length(imdbs{i}.image_ids) == length(roidbs{i}.rois)); % number match
    roidb_merge = [roidb_merge; roidbs{i}.rois'];
    imdb_merge_path = [imdb_merge_path; imdbs{i}.image_ids];
    imdb_merge_size = [imdb_merge_size; imdbs{i}.sizes];
    im_path_root{i} = imdbs{i}.image_dir;
end
imdb_info.roidb_merge = roidb_merge;
imdb_info.imdb_merge_path = imdb_merge_path;
imdb_info.imdb_merge_size = imdb_merge_size;
imdb_info.im_path_root = im_path_root;
end