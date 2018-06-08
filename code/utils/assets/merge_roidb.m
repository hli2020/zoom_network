% deprecated now
num_chunk = 4;
image_roidb = [];

for i = 1:num_chunk

    fprintf('\nchunk %d/%d processing...\n', i, num_chunk);       
    var = matfile(sprintf('./im_roidb_unnormalize_%d.mat', i));
    %curr_image_roidb = var.curr_image_roidb;
    %image_roidb = [image_roidb; curr_image_roidb];
    stats = var.stats;
    image_roidb_part = var.curr_image_roidb(201:400, 1);
    a = 1;
end