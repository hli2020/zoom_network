function startup()
%STARTUP file for the project object proposals using deep models
%   Author:             Hongyang Li
%   Affiliation:        Chinese Univ. of Hong Kong
%   Date:               August, 2016
%   Email:              yangli@ee.cuhk.edu.hk
%   Refactored from:    Shaoqing Ren, Yu Liu.
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'code')));
addpath(genpath(fullfile(curdir, 'experiment')));

mkdir_if_missing(fullfile(curdir, 'code', 'external'));
mkdir_if_missing(fullfile(curdir, 'data', 'datasets'));

mkdir_if_missing(fullfile(curdir, 'output', 'dot_out_file'));
mkdir_if_missing(fullfile(curdir, 'experiment', 'deploy'));

if ~exist('nms_mex', 'file')
    try
        fprintf('Compiling nms_mex\n');

        mex -outdir code/nms ...
          -largeArrayDims ...
          code/nms/nms_mex.cpp ...
          -output nms_mex;
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end
if ~exist('nms_gpu_mex', 'file')
    try
       fprintf('Compiling nms_gpu_mex\n');
       addpath(fullfile(pwd, 'code', 'nms'));
       nvmex('code/nms/nms_gpu_mex.cu', 'code/nms');
       delete('nms_gpu_mex.o');
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end

caffe_path = fullfile(curdir, 'code', 'external', 'caffe', 'matlab');
if exist(caffe_path, 'dir') == 0
    error('matcaffe is missing from external/caffe/matlab; See README.md');
end
addpath(genpath(caffe_path));
caffe.reset_all();
caffe.set_mode_gpu();
clc;

fprintf('caffe version: (%s), cvpr17 startup done!\n', caffe.version);

% TODO: some compilation code here (nms, external box extraction, etc)

