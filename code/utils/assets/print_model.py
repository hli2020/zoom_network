#!/usr/bin/env python2

import argparse
import sys
sys.path.insert(0, '/home/hongyang/project/faster_rcnn/external/caffe/python/caffe/proto')
#import caffe
from caffe_pb2 import NetParameter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            'Copy and rename certain layers into another model.')
    parser.add_argument('model', type=str,
                        help='Source model.')
    args = parser.parse_args()

    source_model = NetParameter.FromString(open(args.model, 'rb').read())

    for layer in source_model.layer:
        print layer.name
