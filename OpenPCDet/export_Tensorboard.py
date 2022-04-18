from tensorboard.backend.event_processing import event_accumulator
import argparse, os

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.split('.')[0] == "events"]

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in_path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex_path', type=str, required=True, help='location to save the exported data')

    args = parser.parse_args()
    gpu = getFiles(args.in_path, "")
    for i in gpu:
        event_data = event_accumulator.EventAccumulator(i)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        if 'Car_3d/easy_R40' in keys:
            car_3d = event_data.scalars.Items('Car_3d/easy_R40')
            car_bev = event_data.scalars.Items('Car_bev/easy_R40')
            # ped_3d = event_data.scalars.Items('Pedestrian_3d/easy_R40')
            # ped_bev = event_data.scalars.Items('Pedestrian_bev/easy_R40')
            cyc_3d = event_data.scalars.Items('Cyclist_3d/easy_R40')
            cyc_bev = event_data.scalars.Items('Cyclist_bev/easy_R40')

            print(i)
            print("car_3d", [i.value for i in car_3d if i.step == 10])
            print("car_bev", [i.value for i in car_bev if i.step == 10])
            print("cyc_3d", [i.value for i in cyc_3d if i.step == 10])
            print("cyc_bev", [i.value for i in cyc_bev if i.step == 10])


if __name__ == '__main__':
    main()