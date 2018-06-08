function [cls, bbox, flag] = convert_ilsvrc_rec( rec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% cls: cell [n x 1], where n is the number of objects
% bbox: double mat [n x 4]
% flag: true if it contains GT in this image

flag = true; cls = []; bbox = [];
if isempty(rec.annotation.object), flag = false; return; end

gt = squeeze(struct2cell(rec.annotation.object));
cls = gt(1, :)';    
bbox = str2double(squeeze(struct2cell(cell2mat(gt(2, :)))))';
bbox = bbox(:, [1 3 2 4]);
end

