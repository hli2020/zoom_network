% make txt file to contain 14 train only

file_path = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit/data/det_lists';
txt_path = dir([file_path '/train_pos_*.txt']);

for i = 1:length(txt_path)
    
    fid = fopen([file_path '/' txt_path(i).name], 'r');
    temp = textscan(fid, '%s');
    im_list = temp{1};
    
    fid2 = fopen([file_path '/' txt_path(i).name(1:end-4) '_hyli.txt'], 'w');
    
    for j = 1:length(im_list)
        
        if strcmp(im_list{j}(1:10), 'ILSVRC2014')
            fprintf(fid2, [im_list{j} '\n']);
        end
    end
    fclose(fid);
    fclose(fid2);
end