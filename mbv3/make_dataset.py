import json
import os

with open("imagenet_class_index.json", 'r') as f:
    load_dict = json.load(f)

class_dict = {}
for item in load_dict:
    class_dict[load_dict[item][0]] = item

ImageNet_Path = './val_img'
dir_list = os.listdir(ImageNet_Path)

for dir_name in dir_list:
    class_index_in_str = class_dict.get(dir_name)
    if class_index_in_str is None:
        continue
    else:
        command = 'mv '+os.path.join(ImageNet_Path, dir_name)+' '+os.path.join(ImageNet_Path, class_index_in_str)
        os.system(command)
