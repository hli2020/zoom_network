function anchors = proposal_generate_anchors(varargin)
% anchors = proposal_generate_anchors(cache_name, varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

ip = inputParser;
% the size of the base anchor
ip.addParameter('base_size',       16,              @isscalar);
% ratio list of anchors
ip.addParameter('ratios',          [0.5, 1, 2],     @ismatrix);
% scale list of anchors
ip.addParameter('scales',          2.^(3:5),        @ismatrix);
ip.parse(varargin{:});
opts = ip.Results;

base_anchor             = [1, 1, opts.base_size, opts.base_size];
ratio_anchors           = ratio_jitter(base_anchor, opts.ratios);
anchors                 = cellfun(@(x) scale_jitter(x, opts.scales), num2cell(ratio_anchors, 2), 'UniformOutput', false);
anchors                 = cat(1, anchors{:});

end

function anchors = ratio_jitter(anchor, ratios)
ratios = ratios(:);

w = anchor(3) - anchor(1) + 1;
h = anchor(4) - anchor(2) + 1;
x_ctr = anchor(1) + (w - 1) / 2;
y_ctr = anchor(2) + (h - 1) / 2;
size = w * h;

size_ratios = size ./ ratios;
ws = round(sqrt(size_ratios));
hs = round(ws .* ratios);

anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

function anchors = scale_jitter(anchor, scales)
scales = scales(:);

w = anchor(3) - anchor(1) + 1;
h = anchor(4) - anchor(2) + 1;
x_ctr = anchor(1) + (w - 1) / 2;
y_ctr = anchor(2) + (h - 1) / 2;

ws = w * scales;
hs = h * scales;

anchors = [x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs - 1) / 2];
end

