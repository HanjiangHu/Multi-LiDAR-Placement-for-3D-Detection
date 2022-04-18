import os
from typing import Dict

import numpy as np
import toml
from tqdm import tqdm

from utils import get_all_points_in_bbox

class POM:
    def __init__(self, random=bool, pom_params=Dict, lidar_params=Dict):
        self.random = random
        self.pom_params = pom_params
        self.lidar_params = lidar_params

    def create_random_data(self):
        x_len = int(self.pom_params["pom_x"] * self.pom_params["scale"])
        y_len = int(self.pom_params["pom_y"] * self.pom_params["scale"])
        z_len = int(self.pom_params["pom_z"] * self.pom_params["scale"])

        occupancy = np.random.rand(x_len, y_len, z_len)
        assert np.all(occupancy < 1)
        assert np.all(occupancy >= 0)

        return occupancy

    def create_data_from_logs(self, path):
        x_len = int(self.pom_params["pom_x"] * self.pom_params["scale"])
        y_len = int(self.pom_params["pom_y"] * self.pom_params["scale"])
        z_len = int(self.pom_params["pom_z"] * self.pom_params["scale"])

        occupancy = np.zeros((x_len, y_len, z_len), dtype=int)

        files = os.listdir(path)
        instances = 0
        frame_list = []
        for file in tqdm(files):
            if file.split(".")[0].split('_')[-1] not in frame_list:
                frame_list.append(file.split(".")[0].split('_')[-1])
        for file in tqdm(files):
            bbox = np.load(f"{path}/{file}")
            bbox[1, :] = - bbox[1,:]
            if  np.all(bbox[0,:] <= 40)  and np.all(bbox[0,:] > 0) and np.all(bbox[1,:] <= 20)  and np.all(bbox[1,:] >= -20) and np.all(bbox[2,:] < 3.5)  and np.all(bbox[2,:] > -0.5):

                points = get_all_points_in_bbox(bbox, self.pom_params["scale"])
                added_occupancy_flag = 0
                for i in range(len(points)):
                    x, y, z = points[i]
                    if (
                        x < occupancy.shape[0]
                        and y < occupancy.shape[1]
                        and z < occupancy.shape[2]
                         and   x >= 0
                            and y >= 0
                            and z >= 0
                    ):
                        occupancy[x, y, z] += 1
                        added_occupancy_flag = 1

                if added_occupancy_flag:
                    instances += 1
        print("Total instances ", instances)
        print("Total frames ", len(frame_list))
        return occupancy, len(frame_list)

    def read_kitti_label(self, path):
        lines = open(path).read().split('\n')
        labels = [l.split(' ') for l in lines]
        for i in labels:
            Height, Width, Length, X, Y, Z = i[8:14]



if __name__ == "__main__":
    print("test pom")
