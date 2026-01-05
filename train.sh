#!/bin/bash
# stage 1
python train.py trainset=kitti valset=kitti model=camliraft ckpt.path=checkpoint/vismoflow_initial.pt

# stage 2
python train.py trainset=kitti_event valset=kitti_event model=vismoflow ckpt.path=checkpoint/vismoflow_evkitti.pt

# stage 3
python train.py trainset=dsec valset=dsec model=vismoflow ckpt.path=checkpoint/vismoflow_dsec.pt