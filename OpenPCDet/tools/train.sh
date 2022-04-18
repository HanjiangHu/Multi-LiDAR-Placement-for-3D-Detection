export CUDA_HOME=/usr/local/cuda
cd ./OpenPCDet/tools
python train.py --cfg_file cfgs/kitti_models/voxel_rcnn.yaml --batch_size 44 --pretrained_model voxel_rcnn_car_84.54.pth
python train.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --batch_size 12 --pretrained_model pointrcnn_iou_7875.pth
python train.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 12 --pretrained_model pointrcnn_7870.pth
python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --pretrained_model pv_rcnn_8369.pth
python train.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 80 --pretrained_model second_7862.pth
python train.py --cfg_file cfgs/kitti_models/second_iou.yaml --batch_size 80 --pretrained_model second_iou7909.pth
