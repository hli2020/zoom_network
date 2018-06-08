function [cls, bbox, flag] = convert_pascal_rec( rec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% cls: cell [n x 1], where n is the number of objects
% bbox: double mat [n x 4]
% flag: true if it contains GT in this image

flag = true; cls = []; bbox = [];
if isempty(rec.objects), flag = false; return; end

gt = squeeze(struct2cell(rec.objects));
cls = gt(1, :)'; % cell [gt_num x 1]
bbox = cell2mat(gt(8, :));
bbox = reshape(bbox, [4 length(cls)])';

class_list = {'aeroplane','bicycle','bird','boat',...
'bottle','bus','car','cat',...
'chair','cow','diningtable','dog',...
'horse','motorbike','person','pottedplant',...
'sheep','sofa','train','tvmonitor'};
cls = cellfun(@(x) find(strcmp(x, class_list)), cls);
end

