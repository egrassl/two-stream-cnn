import os
import utils.file_management as fm
import dataset_tools.ntu_definitions as ntu


args = {}


def on_class_load_spatial(videos, path, class_name):

    # Creates destiny folders if doesnt exists
    fm.create_dirs([
        os.path.join(args['s_dest_train'], class_name),
        os.path.join(args['s_dest_val'], class_name),
        os.path.join(args['s_dest_test'], class_name)
    ], args['verbose'])


def on_class_load_temporal(videos, path, class_name, mode):
    # Creates destiny folders if doesnt exists
    if mode == 'u':
        fm.create_dirs([
            os.path.join(args['t_dest_train_u'], class_name),
            os.path.join(args['t_dest_val_u'], class_name),
            os.path.join(args['t_dest_test_u'], class_name)
        ], args['verbose'])

    elif mode == 'v':
        fm.create_dirs([
            os.path.join(args['t_dest_train_v'], class_name),
            os.path.join(args['t_dest_val_v'], class_name),
            os.path.join(args['t_dest_test_v'], class_name)
        ], args['verbose'])


def get_indexes(files):
    files.sort()

    # Calculate indexes
    total = len(files)
    train_index = int(total * (1 - args['validation'] - args['test']))
    val_index = int(total * (1 - args['test']))
    test_index = total

    return train_index, val_index, test_index


def copy_files(train, val, test, class_name, mode):
    # Copy all train, validation and test samples for the specified mode
    if mode == 'spatial':
        fm.copy_files(train, os.path.join(args['s_dest_train'], class_name), args['verbose'])
        fm.copy_files(val, os.path.join(args['s_dest_val'], class_name), args['verbose'])
        fm.copy_files(test, os.path.join(args['s_dest_test'], class_name), args['verbose'])

    elif mode == 'u':
        fm.copy_files(train, os.path.join(args['t_dest_train_u'], class_name), args['verbose'])
        fm.copy_files(val, os.path.join(args['t_dest_val_u'], class_name), args['verbose'])
        fm.copy_files(test, os.path.join(args['t_dest_test_u'], class_name), args['verbose'])

    elif mode == 'v':
        fm.copy_files(train, os.path.join(args['t_dest_train_v'], class_name), args['verbose'])
        fm.copy_files(val, os.path.join(args['t_dest_val_v'], class_name), args['verbose'])
        fm.copy_files(test, os.path.join(args['t_dest_test_v'], class_name), args['verbose'])
    else:
        raise Exception('There is not a file mode definition for \'%s\'' % mode)


def split_spatial_frames(files, path, class_name):
    train_index, val_index, test_index = get_indexes(files)
    copy_files(files[:train_index], files[:train_index], files[val_index:test_index], class_name, 'spatial')


def split_temporal_frames(files, path, class_name, mode):
    train_index, val_index, test_index = get_indexes(files)
    copy_files(files[:train_index], files[train_index:val_index], files[val_index:test_index], class_name, mode)


def ntu_cross_subject_spatial(files, path, class_name):
    train, validation, test = ntu.get_cs_split(files)
    copy_files(train, validation, test, class_name, 'spatial')


def ntu_cross_subject_temporal(files, path, class_name, mode):
    train, validation, test = ntu.get_cs_split(files)
    copy_files(train, validation, test, class_name, mode)


def ntu_cross_view_spatial(files, path, class_name):
    train, validation, test = ntu.get_cv_split(files)
    copy_files(train, validation, test, class_name, 'spatial')


def ntu_cross_view_temporal(files, path, class_name, mode):
    train, validation, test = ntu.get_cv_split(files)
    copy_files(train, validation, test, class_name, mode)

def init_directories():
    # Get spatial destiny paths
    if args['spatial']:
        args['s_dest_train'] = os.path.join(args['dst'], 'spatial', 'train')
        args['s_dest_val'] = os.path.join(args['dst'], 'spatial', 'val')
        args['s_dest_test'] = os.path.join(args['dst'], 'spatial', 'test')

        # creates spatial destiny paths if they dont exist
        fm.create_dirs([
            os.path.join(args['dst'], 'spatial'),
            args['s_dest_train'],
            args['s_dest_val'],
            args['s_dest_test']
        ], args['verbose'])

    # Get temporal destiny paths
    if args['temporal']:
        # horizontal flow
        args['t_dest_train_u'] = os.path.join(args['dst'], 'temporal', 'train', 'u')
        args['t_dest_val_u'] = os.path.join(args['dst'], 'temporal', 'val', 'u')
        args['t_dest_test_u'] = os.path.join(args['dst'], 'temporal', 'test', 'u')

        # vertical flow
        args['t_dest_train_v'] = os.path.join(args['dst'], 'temporal', 'train', 'v')
        args['t_dest_val_v'] = os.path.join(args['dst'], 'temporal', 'val', 'v')
        args['t_dest_test_v'] = os.path.join(args['dst'], 'temporal', 'test', 'v')

        # creates temporal destiny paths if they dont exist
        fm.create_dirs([
            args['t_dest_train_u'],
            args['t_dest_val_u'],
            args['t_dest_test_u'],
            args['t_dest_train_v'],
            args['t_dest_val_v'],
            args['t_dest_test_v']
        ], args['verbose'])


def split_dataset(src, dst, validation, test, spatial, temporal, mode='split', verbose=False):
    # Sets arguments
    args['src'] = src
    args['dst'] = dst
    args['validation'] = validation
    args['test'] = test
    args['spatial'] = spatial
    args['temporal'] = temporal
    args['verbose'] = verbose

    init_directories()

    # For spatial
    if args['spatial']:
        if mode == 'split':
            fm.class_iterator(path=src,
                              on_class_loaded=on_class_load_spatial,
                              process_sample=None,
                              on_class_finished=split_spatial_frames,
                              extension='.jpg',
                              verbose=args['verbose'])
        elif mode == 'ntu-cs':
            fm.class_iterator(path=src,
                              on_class_loaded=on_class_load_spatial,
                              process_sample=None,
                              on_class_finished=ntu_cross_subject_spatial,
                              extension='.jpg',
                              verbose=args['verbose'])
        elif mode == 'ntu-cv':
            fm.class_iterator(path=src,
                              on_class_loaded=on_class_load_spatial,
                              process_sample=None,
                              on_class_finished=ntu_cross_view_spatial,
                              extension='.jpg',
                              verbose=args['verbose'])
        else:
            raise Exception('There is not a split mode definition for \'%s\'' % mode)

    # For temporal
    if args['temporal']:
        if mode == 'split':
            fm.flow_class_iterator(path=src,
                                   on_class_loaded=on_class_load_temporal,
                                   process_video=None,
                                   on_class_finished=split_temporal_frames,
                                   extension='.jpg',
                                   verbose=args['verbose'])
        elif mode == 'ntu-cs':
            fm.flow_class_iterator(path=src,
                                   on_class_loaded=on_class_load_temporal,
                                   process_video=None,
                                   on_class_finished=ntu_cross_subject_temporal,
                                   extension='.jpg',
                                   verbose=args['verbose'])
        elif mode == 'ntu-cv':
            fm.flow_class_iterator(path=src,
                                   on_class_loaded=on_class_load_temporal,
                                   process_video=None,
                                   on_class_finished=ntu_cross_view_temporal,
                                   extension='.jpg',
                                   verbose=args['verbose'])
        else:
            raise Exception('There is not a split mode definition for \'%s\'' % mode)
