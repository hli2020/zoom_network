function conf= init_zoom_net(conf)
% --------------------------------------------------------
% Zoom Network
% Copyright (c) 2017, Hongyang Li
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

% adjust the configuration
% || prepare anchors
if conf.multi_depth
    m_level = 3;
    conf.anchors = proposal_generate_anchors_multi(...
        'base_size',    conf.base_size, ...
        'scales',       conf.anchor_scale, ...
        'ratios',       conf.ratios);
    
else
    m_level = 1;
    conf.anchors{1} = proposal_generate_anchors(...
        'base_size',    conf.base_size, ...
        'scales',       conf.anchor_scale, ...
        'ratios',       conf.ratios ...
        );
end

if ~conf.check_certain_scale
    output_map_cache = sprintf('data/output_map_min%d_max%d_everScale.mat', ...
        conf.min_size, conf.max_size);
else
    output_map_cache = sprintf('data/output_map_min%d_max%d_certScale.mat', ...
        conf.min_size, conf.max_size);
end

% || traversal search on possible output size of the network
try
    ld = load(output_map_cache);
    conf.output_width_map = ld.output_width_map;
    conf.output_height_map = ld.output_height_map;
    if conf.check_certain_scale
        conf.certain_scale = ld.certain_scale; 
    end
    clear ld;
catch
    cprintf('blue', 'pre-calculating output size of the zoom network...\n');
    gpu_id = conf.gpu_id(1);
    solver_file = 'model/zoom/bn/base/solver_deploy_check.prototxt';
    
    if conf.check_certain_scale
        % for zoom network (hourglass) structure
        [conf.output_width_map, conf.output_height_map, conf.certain_scale] = ...
            proposal_calc_output_size_certainScale(conf, ...
            solver_file, gpu_id, length(conf.rpn_feat_stride));        
        % save the file for future use
        output_width_map = conf.output_width_map;
        output_height_map = conf.output_height_map;
        certain_scale = conf.certain_scale;
        save(output_map_cache, 'output_width_map', 'output_height_map', 'certain_scale');
    else
        assert(m_level==1);
        output_name = 'inception_5b/output';
        for i = 1:m_level
            [conf.output_width_map{i}, conf.output_height_map{i}] = ...
                proposal_calc_output_size(conf, ...
                solver_file, gpu_id, output_name, i);
        end
        output_width_map = conf.output_width_map;
        output_height_map = conf.output_height_map;
        save(output_map_cache, 'output_width_map', 'output_height_map');
    end
end

if strcmp(conf.mode, 'train')
    assert(conf.fg_thresh > conf.gray_hi);
    assert(conf.gray_hi > conf.gray_lo);
    assert(conf.gray_lo > conf.bg_thresh_hi);
end
if strcmp(conf.mode, 'test')
    conf.chunk_mode = true;
    try
        conf.total_chunk;
        conf.curr_chunk;
    catch
        conf.chunk_mode = false;
    end
end
