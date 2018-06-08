import json
import math

root = '/home/hongyang/project/externalBox/data_I_want_coco/'
data_name = root + 'deepmask-coco-val-bbox/deepmaskZoom_bbox_coco_val.json'
chunk_num = 50000.0

with open(data_name) as data_file:
    data = json.load(data_file)
# data = range(15)
iter_length = int(math.ceil(len(data)/chunk_num))

for i in range(iter_length):

    start_id = int(i*chunk_num)
    end_id = int(min(chunk_num + i*chunk_num, len(data)))
    curr_data = data[start_id:end_id]

    save_name = 'split_ck_%d_total_%d.json' % (i+1, iter_length)
    with open(save_name, 'w') as outfile:
        json.dump(curr_data, outfile, ensure_ascii=False)

    print 'process iter: %d, total: %d\n' (i+1, iter_length)
