# Multi-LiDAR-Placement-for-3D-Detection
This is the official released code for CVPR 2022 "Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving". Check out [arXiv PDF](https://arxiv.org/abs/2105.00373) for more details.


## Preparation
Install and set up all the environments of [CARLA v0.9.10](https://carla.readthedocs.io/en/0.9.10/), [ScenarioRunner](https://github.com/carla-simulator/scenario_runner), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) of commit `cbf2f4e` on Ubuntu 18.04.



## Data Collection
Fill in the CARLA related absolute paths on `Line 36-49` in `./carla-kitti/scenario_runner/scenario_runner.py`.
Turn on CARLA as default and then run the following shell script. 

`cd ./carla-kitti`

`bash routes_baselines.sh`

Different LiDAR configurations can be accessed at `./carla-kitti/hyperparams`. The density of Vehicles and Pedestrians can be changed on `Line 397-422` in `./carla-kitti/scenario_runner/srunner/scenarios/route_scenario.py`.


## Model Training and Testing
To split the training and test set for the experiment, copy  `./OpenPCDet/tools/split_training_test.py` to the root path of the collected dataset (e.g. `./carla-kitti/dataset/center/`) and run it under that path. Then filter the useless bounding boxes labels in KITTI format through the following command, e.g.

`python ./OpenPCDet/tools/filter_label.py -r ./carla-kitti/dataset/center/`

Follow the instructions of OpenPCDet to move the KITTI-formatted CARLA dataset to `./OpenPCDet/data/kitti/` and run the preparation script for OpenPCDet,

`python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`

Download the KITTI pretrained models and fine-tune and test them on our newly collected dataset from CARLA using specific LiDAR configurations. 

`bash ./OpenPCDet/tools/train.sh`

`bash ./OpenPCDet/tools/test.sh`

## Surrogate Metric Validation
To evaluate the LiDAR placement, change the directory `cd ./S_MIG` and run the command,

`python evaluate_lidar_position.py`

Make sure that POG in `./S_MIG/routes/` is not empty after the data collection step and the default POG is calculated under Square placement.



### Reference
> - [CARLA v0.9.10](https://carla.readthedocs.io/en/0.9.10/)
> - [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
> - [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) of commit `cbf2f4e`

### Citation
If you use this code in your own work, please cite this paper:

H. Hu*, Z. Liu*, S. Chitlangia, A. Agnihotri and D. Zhao
"[Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving](https://arxiv.org/abs/2105.00373)", CVPR 2022

```
@article{hu2021investigating,
  title={Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving},
  author={Hu, Hanjiang and Liu, Zuxin and Chitlangia, Sharad and Agnihotri, Akhil and Zhao, Ding},
  journal={arXiv e-prints},
  pages={arXiv--2105},
  year={2021}
}
```