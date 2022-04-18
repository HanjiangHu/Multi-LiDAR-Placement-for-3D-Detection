from shutil import move
import glob, os

original_path = "training"
all_paths = sorted(glob.glob(original_path+'/*'))
if not os.path.exists("ImageSets"):
    os.makedirs("ImageSets")
write_txt_flag = 1
for path_item in all_paths:
    items = sorted(glob.glob(path_item + '/*'))
    target_path_item = path_item.replace('training', 'testing')
    if not os.path.exists(target_path_item):
        os.makedirs(target_path_item)
    for index, item in enumerate(items):
        if index % 10 == 9:
            move(item, target_path_item+'/'+item.split('/')[-1])
            if write_txt_flag:
                with open("ImageSets/test.txt","a") as f:
                    f.write(item.split('/')[-1].split('.')[-2])
                    f.write('\n')
                with open("ImageSets/val.txt","a") as f:
                    f.write(item.split('/')[-1].split('.')[-2])
                    f.write('\n')
        else:
            if write_txt_flag:
                with open("ImageSets/train.txt","a") as f:
                    f.write(item.split('/')[-1].split('.')[-2])
                    f.write('\n')
    write_txt_flag = 0



