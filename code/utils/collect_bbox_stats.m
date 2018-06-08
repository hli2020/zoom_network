function conf = collect_bbox_stats(opts, conf)
% INACTIVE for now
data_path = fullfile(pwd, 'data/training_test_data/rpn');
if ~isempty(opts.share_data_name)
    train_data_file = fullfile(data_path, opts.share_data_name, 'train.mat');
else
    train_data_file = fullfile(data_path, conf.model_id, 'train.mat');
end
mkdir_if_missing(fileparts(train_data_file));
train_data_name_lite = strrep(train_data_file, pwd, '');
% train data
if exist(train_data_file, 'file') && opts.detect_exist_train_file
    
    fprintf('|| Loading existant stats of RPN training data (%s) ...\n', ...
        train_data_name_lite);
    ld = load(train_data_file);
    bbox_means = ld.bbox_means; bbox_stds = ld.bbox_stds;
    clear ld; fprintf('   Done.\n');
else
    
    fprintf('|| Computing stats of RPN training data (%s) ...\n', ...
        train_data_name_lite);
    [bbox_means, bbox_stds]...
        = proposal_prepare_image_roidb_on_the_fly(...
        conf.rpn_param, opts.imdb_train, opts.roidb_train);
    save(train_data_file, 'bbox_means', 'bbox_stds', '-v7.3');
    fprintf(' Done and saved.\n\n');
end
conf.rpn_param.bbox_means = bbox_means;
conf.rpn_param.bbox_stds = bbox_stds;
end

