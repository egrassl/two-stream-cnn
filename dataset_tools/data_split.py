import os
import glob
import utils.file_management as fm
import shutil


def copy_videos_from_path(videos, dst, verbose=False):
    '''
    Copy videos from a path to another path

    :param videos: list with videos path
    :param dst: destiny path
    '''
    for v in videos:
        path, video_name = os.path.split(v)
        shutil.copyfile(v, os.path.join(dst, video_name))

        if verbose:
            print("Video copied from %s to %s" % (v, os.path.join(dst, video_name)))


def split_spatial(src, dest, validation, test, verbose=False):
    '''
    Copy videos from spatial folder of dataset
    :param src:
    :param dest:
    :param validation:
    :param test:
    :param verbose:
    :return:
    '''

    # Get original dataset spatial path
    path_spatial = os.path.join(src, 'spatial')

    # Get destiny paths
    dest_train = os.path.join(dest, 'spatial', 'train')
    dest_val = os.path.join(dest, 'spatial', 'val')
    dest_test = os.path.join(dest, 'spatial', 'test')

    # create destiny paths if they dont exist
    fm.create_dirs([
        os.path.join(dest, 'spatial'),
        dest_train,
        dest_val,
        dest_test
    ], verbose)

    # Get video classes
    classes = glob.glob(os.path.join(path_spatial, '*'))
    if verbose:
        print('%d classes detected' % len(classes))

    for c in classes:

        # Get class name
        class_name = c.split('/')[-1]

        # Creates destiny folders if doesnt exists
        fm.create_dirs([
            os.path.join(dest_train, class_name),
            os.path.join(dest_val, class_name),
            os.path.join(dest_test, class_name)
        ], verbose)

        if verbose:
            print('\nClass %s\n==================\n' % class_name)

        # Get all videos for class c
        videos = glob.glob(os.path.join(c, '*.jpg'))

        # Calculate indexes
        total = len(videos)
        train_index = int(total * (1 - validation - test))
        val_index = int(total * (1 - test))
        test_index = total

        # Copy all train videos
        copy_videos_from_path(videos[:train_index], os.path.join(dest_train, class_name), verbose)

        # Copy all validation videos
        copy_videos_from_path(videos[train_index:val_index], os.path.join(dest_val, class_name), verbose)

        # Copy all test videos
        copy_videos_from_path(videos[val_index:test_index], os.path.join(dest_test, class_name), verbose)
