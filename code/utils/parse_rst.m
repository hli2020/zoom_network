function results = parse_rst(results, rst_net, iter, debug)
% from Liu Yu, adapted to multi-gpu training
% average the results across different cards

rst_singleNet = rst_net{1};     % first net result
rst_singleNet = changeNanToZero(rst_singleNet, iter, 1, debug);

% i is the net index, j is the data index per net
for i = 2 : length(rst_net)
    
    rst_net{i} = changeNanToZero(rst_net{i}, iter, i, debug);
    for j = 1 : length(rst_singleNet)
        rst_singleNet(j).data = rst_singleNet(j).data + rst_net{i}(j).data;
    end
end

for j = 1 : length(rst_singleNet)
    rst_singleNet(j).data = rst_singleNet(j).data ./ length(rst_net);
end

if isempty(results)
    for j = 1:length(rst_singleNet)
        results.(rst_singleNet(j).blob_name).data = [];
    end
end

for j = 1:length(rst_singleNet)
    results.(rst_singleNet(j).blob_name).data = ...
        [results.(rst_singleNet(j).blob_name).data; rst_singleNet(j).data(:)];
end
end

function rst_singleNet = changeNanToZero(rst_singleNet, iter, which_net, debug)
raw_data = extractfield(rst_singleNet, 'data');
if debug
    if any(isnan(raw_data))
        cprintf('blue', 'abs iter:\t%d, which_gpu: #%d, NaN result occurs. change to ZERO just in matlab GUI.\n', ...
            iter, which_net);
    end
end
raw_data(isnan(raw_data)) = 0;
for kk = 1:length(rst_singleNet)
    rst_singleNet(kk).data = raw_data(kk);
end
end
