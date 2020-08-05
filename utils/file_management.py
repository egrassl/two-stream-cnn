import os
import glob
import utils.video_processing as vp
import shutil


def copy_files(files, dst, verbose=False):
    '''
    Copy videos from a path to another path

    :param files: list with videos path
    :param dst: destiny path
    '''
    for f in files:
        path, file_name = os.path.split(f)
        shutil.copyfile(f, os.path.join(dst, file_name))

        if verbose:
            print("File copied from %s to %s" % (f, os.path.join(dst, file_name)))


def create_dir(path, verbose=False):
    '''
    Creates a directory if it doesn't already exists

    :param path: path to directory
    :param verbose: prints when the directory is created if it is set to True
    :return: True if directory was created and False if it already existed
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        if verbose:
            print('Directory %s created' % path)
        return True
    else:
        return False


def create_dirs(paths, verbose=False):
    '''
    Try to create directories specified by a list and returns True if one or more directory was created

    :param paths: list of directories to create
    :param verbose: prints when a directory is created if it is set to True
    :return: returns True if one or more directories were created and False otherwise
    '''
    created = False
    for path in paths:
        created = create_dir(path, verbose) or created

    return created


def class_iterator(path, process_sample, on_class_loaded, on_class_finished, extension='.avi', verbose=False):
    # Get samples classes
    classes = glob.glob(os.path.join(path, '*'))
    if verbose:
        print('%d classes detected' % len(classes))

    # Iterates trough each class
    for c in classes:

        # Gets files inside class folder
        videos = glob.glob(os.path.join(c, '*%s' % extension))

        _, class_name = os.path.split(c)

        if on_class_loaded is not None:
            on_class_loaded(videos, c, class_name)

        if verbose:
            print('\nClass %s\n==================\n' % class_name)

        # Load all frames and process video if specified (only if a processing function is given)
        if process_sample is not None:
            for v_path in videos:
                frames = vp.read_video(v_path)

                _, video_name = os.path.split(v_path)

                process_sample(frames, video_name, c, class_name)

        if on_class_finished is not None:
            on_class_finished(videos, c, class_name)
