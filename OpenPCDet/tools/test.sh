export CUDA_HOME=/usr/local/cuda
cd ./OpenPCDet/tools
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 6 --ckpt ../../output/kitti_models/pv_rcnn/default/ckpt/checkpoint_epoch_10.pth &>> ./test_pvrcnn.out
python test.py --cfg_file cfgs/kitti_models/voxel_rcnn.yaml --batch_size 6 --ckpt ../../output/kitti_models/voxel_rcnn/default/ckpt/checkpoint_epoch_10.pth &>> ./test_voxel_rcnn.out
python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 6 --ckpt ../../output/kitti_models/pointrcnn/default/ckpt/checkpoint_epoch_10.pth &>> ./test_pointrcnn.out
python test.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --batch_size 6 --ckpt ../../output/kitti_models/pointrcnn_iou/default/ckpt/checkpoint_epoch_10.pth &>> ./test_pointrcnn_iou.out
python test.py --cfg_file cfgs/kitti_models/second_iou.yaml --batch_size 6 --ckpt ../../output/kitti_models/second_iou/default/ckpt/checkpoint_epoch_10.pth &>> ./test_second_iou.out
python test.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 6 --ckpt ../../output/kitti_models/second/default/ckpt/checkpoint_epoch_10.pth &>> ./test_second.out
