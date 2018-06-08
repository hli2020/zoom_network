function imdb_info = get_coco_info(imdbs)
imdb_info.im_id = [];
imdb_info.imdb_merge_size = [];
imdb_info.imdb_merge_path = [];

for i = 1:length(imdbs)
    curr_struct = imdbs{i}.data.images;
    curr_size_mat = [extractfield(curr_struct, 'height')' extractfield(curr_struct, 'width')'];
    imdb_info.imdb_merge_size = [imdb_info.imdb_merge_size; curr_size_mat];
    
    imdb_info.im_id = [imdb_info.im_id; extractfield(curr_struct, 'id')'];
    imdb_info.imdb_merge_path = [imdb_info.imdb_merge_path; extractfield(curr_struct, 'file_name')'];
end
end