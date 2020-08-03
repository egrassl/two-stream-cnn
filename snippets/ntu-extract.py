import glob
import os
import dataset_tools.ntu_definitions as ntu
import utils.file_management as fm

src = '/home/coala/mestrado/datasets/NTU-HOME/raw'
dst = '/home/coala/mestrado/datasets/NTU-HOME/raw'

videos = glob.glob(os.path.join(src, '*.avi'))

class_folders = [os.path.join(dst, class_name) for class_name in ntu.classes.values()]

fm.create_dirs(class_folders, True)

for v in videos:
    path, video_name = os.path.split(v)
    destiny = os.path.join(dst, ntu.classes[video_name[16:20]])
    fm.copy_files([v], destiny, True)
