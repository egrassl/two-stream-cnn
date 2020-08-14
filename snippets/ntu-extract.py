import glob
import os
import dataset_tools.ntu_definitions as ntu
import utils.file_management as fm

src = r'D:\Mestrado\code\databases\NTU\videos'
dst = r'D:\Mestrado\code\databases\NTU\videos-classes'

videos = glob.glob(os.path.join(src, '*.avi'))

# class_folders = [os.path.join(dst, class_name) for class_name in ntu.classes.values()]

class_folders = [os.path.join(dst, class_name) for class_name in ntu.classes_all.values()]

fm.create_dirs(class_folders, True)

for v in videos:
    path, video_name = os.path.split(v)
    # destiny = os.path.join(dst, ntu.classes[video_name[16:20]])
    destiny = os.path.join(dst, ntu.classes_all[video_name[16:20]])
    fm.copy_files([v], destiny, True)
