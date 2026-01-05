from __future__ import print_function, division
import sys
# sys.path.append('core')
import math
import os
import numpy as np 

training_name = 'training.txt'
testing_name = 'testing.txt'
root_dir = 'G:/CVPR_DSEC/Nighttime'
PATH = os.listdir(root_dir)
# 遍历根目录下的所有文件夹
with open(os.path.join(root_dir, testing_name), "w") as testing_f:
    with open(os.path.join(root_dir, training_name), "w") as traning_f:
        for dir_name in PATH:
            # 组装当前文件夹的完整路径
            current_dir_path = os.path.join(root_dir, dir_name)
            # image_path = os.path.join(dir_name, 'images_rectify')
            # event_path = os.path.join(dir_name, 'event_voxel')
            # flow_path = os.path.join(dir_name, 'flow')
            files = sorted(os.listdir(os.path.join(current_dir_path, 'flow')))
            num = len(files)
            testing_num = np.random.randint(0, num, size=int(0.1*num))
            with open(os.path.join(current_dir_path, 'forward_timestamps.txt'), 'r') as f:
                timestamps = f.readlines()
                del timestamps[0]
                f.close()

            for idx in range(num):
                frame1_id = int(files[idx].split('.')[0])
                frame1_id = str(frame1_id).zfill(6)
                frame2_id = int(files[idx].split('.')[0]) + 2
                frame2_id = str(frame2_id).zfill(6)
                t_1 = timestamps[idx].split(', ')[0]
                t_2 = timestamps[idx].split(', ')[1].split('\n')[0]
                line = dir_name + ' ' + frame1_id + ' ' + frame2_id + ' ' + t_1 + ' ' + t_2 + '\n'
                line = line.replace('\\', '/')
                if idx in testing_num:
                    testing_f.write(line)
                else:
                    traning_f.write(line)
        
        traning_f.close()
        testing_f.close()

