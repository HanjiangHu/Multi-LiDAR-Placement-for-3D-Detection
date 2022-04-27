import toml
import argparse
import numpy as np

from scipy.stats import entropy

from pom import POM
from sample_script import get_points_covered_by_lidar_config

def evaluate(map, pom_params, lidar_params, config):
        points = get_points_covered_by_lidar_config(
            pom_params, lidar_params, config,  lidar_params['lidar_nos']
        )

        H_entropy = 0.0 # -plogp-(1-p)log(1-p)

        total_entropy = 0.0
        for x in map:
            for xy in x:
                for xyz in xy:
                    if xyz == 0.0 or xyz == 1:
                        continue
                    total_entropy += entropy([1 - xyz, xyz])

        for point in range(len(points)):
            p = map[points[point]]
            if p == 0.0 or p == 1:
                continue
            H_entropy += entropy([1-p,p])

        return H_entropy, total_entropy, total_entropy - H_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--params', type=str, default="multihyper.toml", help="Params")
    parser.add_argument('-c','--configuration', type=str, default="config.toml", help="Configuration")

    args = parser.parse_args()
    params = toml.load(args.params)
    configs = toml.load(args.configuration)['config']

    pom_car, num_valid_frames_car = POM(
        random=True, pom_params=params["pom"], lidar_params=params["lidar"]
    ).create_data_from_logs(
        "./routes/square/vehicle"
    )
    print(111)
    pom_car = pom_car.astype(float) / num_valid_frames_car

    pom_ped, num_valid_frames_ped = POM(
        random=True, pom_params=params["pom"], lidar_params=params["lidar"]
    ).create_data_from_logs(
        "./routes/square/pedestrian"
    )

    pom_ped = pom_ped.astype(float) / num_valid_frames_ped

    pom_cyc, num_valid_frames_cyc = POM(
        random=True, pom_params=params["pom"], lidar_params=params["lidar"]
    ).create_data_from_logs(
        "./routes/square/cyclist"
    )

    pom_cyc = pom_cyc.astype(float) / num_valid_frames_cyc
    type = ['square', 'center', 'line', 'pyramid', 'trapezoid', 'line_roll', 'pyramid_roll',
            'pyramid_pitch']
    pom_list = [('car', pom_car), ('ped', pom_ped), ('cyc', pom_cyc)]
    for key, config in configs.items():
        for pom in pom_list:
            H_entropy, total_entropy, IG = evaluate(pom[1], params['pom'], params['lidar'], config)
            print(
                f"Key {type[int(key)]}, {pom[0]}: H_entropy {H_entropy}, total_entropy {total_entropy}, IG {IG}")