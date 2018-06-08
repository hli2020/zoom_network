% compute coco image mean

train_set = './data/datasets/coco/train2014';
val_set = './data/datasets/coco/val2014';

ld = load('./model/pretrain/coco_mean_image_train.mat');
train_rgb = ld.train_rgb;

% train_list = dir([train_set '/*.jpg']);
% train_rgb = zeros(length(train_list), 3);
% parfor i = 1:length(train_list)
%     im = imread([train_set '/' train_list(i).name]);
%     if size(im, 3) == 1
%         im = repmat(im, [1 1 3]);
%     end
%     train_rgb(i, :) = [mean(mean(im(:, :, 1))) ...
%         mean(mean(im(:, :, 2))) mean(mean(im(:, :, 3)))];
% end
% disp(mean(train_rgb));
% save('./model/pretrain/coco_mean_image_train.mat', 'train_rgb');

val_list = dir([val_set '/*.jpg']);
val_rgb = zeros(length(val_list), 3);

parfor i = 1:length(val_list)
    im = imread([val_set '/' val_list(i).name]);
    if size(im, 3) == 1
        im = repmat(im, [1 1 3]);
    end
    val_rgb(i, :) = [mean(mean(im(:, :, 1))) ...
        mean(mean(im(:, :, 2))) mean(mean(im(:, :, 3)))];
end
disp(mean(val_rgb));
save('./model/pretrain/coco_mean_image_val.mat', 'val_rgb');

image_mean = mean([train_rgb; val_rgb]);
image_mean = single(reshape(image_mean, [1 1 3]));
save('./model/pretrain/coco_mean_image.mat', 'image_mean');
