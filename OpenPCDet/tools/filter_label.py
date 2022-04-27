import shutil 
import glob, os, argparse


parser = argparse.ArgumentParser()

parser.add_argument('-r','--root_path', type=str, default="", help="root_path")


args = parser.parse_args()

root_path = args.root_path
original_path = root_path + "/testing"
shutil.copytree(original_path,original_path.replace('testing', '360_testing_'))
shutil.move(root_path + "/ImageSets/val.txt", original_path.replace('testing', '360_testing_') + "/val.txt")
shutil.move(root_path + "/ImageSets/test.txt", original_path.replace('testing', '360_testing_') + "/test.txt")
label_path = original_path + '/label_2'
all_paths = sorted(glob.glob(label_path+'/*'))
new_label_path = label_path.replace('label_2','label_2_180')
if not os.path.exists(new_label_path):
    os.makedirs(new_label_path)
write_txt_flag = 1
for txt_item in all_paths:
    with open(txt_item, "r") as f:
        for line in f.readlines():
            z = float(line.split(' ')[13])
            print(z)
            if z <= 0: 
                continue
            print(line)
            with open(new_label_path+'/'+txt_item.split('/')[-1],"a") as f:
                f.write(line)
os.rename(label_path, label_path.replace('label_2','label_2_360'))
os.rename(new_label_path, new_label_path.replace('label_2_180','label_2'))
shutil.rmtree(label_path.replace('label_2','label_2_360'))


all_paths = sorted(glob.glob(original_path+'/*'))
label_items = sorted(glob.glob(label_path+'/*'))
for path in all_paths:
    all_items = sorted(glob.glob(path+'/*'))
    for item in all_items:
        item_index_old = item.split('/')[-1].split('.')[-2]
        item_old_label = label_path + '/' + item_index_old + '.txt'
        if item_old_label not in label_items:
            os.remove(item)

for item in label_items:
    with open(root_path + "/ImageSets/test.txt","a") as f:
        f.write(item.split('/')[-1].split('.')[-2])
        f.write('\n')   
    with open(root_path + "/ImageSets/val.txt","a") as f:
        f.write(item.split('/')[-1].split('.')[-2])
        f.write('\n')
if os.path.exists(root_path+'/kitti_dbinfos_train.pkl'):
    os.remove(root_path+'/kitti_dbinfos_train.pkl')
if os.path.exists(root_path+'/kitti_infos_test.pkl'):
    os.remove(root_path+'/kitti_infos_test.pkl')
if os.path.exists(root_path+'/kitti_infos_train.pkl'):
    os.remove(root_path+'/kitti_infos_train.pkl')
if os.path.exists(root_path+'/kitti_infos_val.pkl'):
    os.remove(root_path+'/kitti_infos_val.pkl')
if os.path.exists(root_path+'/gt_database'):
    shutil.rmtree(root_path+'/gt_database')

