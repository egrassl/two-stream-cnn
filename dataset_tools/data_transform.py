import os
import cv2
import utils.file_management as fm
import utils.video_processing as vp

# Arguments dictionary
args = {}


def extract_video(frames, video_name, class_path, class_name):

    print('\n(%d, %d, %d) video loaded: %s' % (len(frames), len(frames[0]), len(frames[0][1]), video_name))

    # Finds T equally distributed indexes to represent the T video chunks
    step = float((len(frames) - 1) / (args['chunks'] + 1))
    indexes = [round(step * i) for i in range(1, args['chunks'] + 1)]

    count = 0
    video_rpath = os.path.join(class_name, video_name)
    for i in indexes:

        # Gets video file count and relative path
        suffix = str(count).zfill(3)

        # get spatial frames
        if args['spatial']:
            file_name = video_rpath[:-4] + '_%s.jpg' % suffix
            name = os.path.join(args['path_spatial'], file_name)
            cv2.imwrite(name, vp.frame_resize(frames[i]))
            if args['verbose']:
                print('File created: %s' % name)

        # Gets optical flow frames
        if args['temporal']:

            # Get stacked frames
            stacked_frames = vp.extract_motion_from_frames(frames, i - args['nb_range'], i + args['nb_range'],
                                                           args['nb_range'], width=350)

            # Saves horizontal
            for j in range(0, args['nb_range']):
                video_name = video_rpath[:-4] + '_%s_%s.jpg' % (suffix, str(j).zfill(2))

                # Saves horizontal flow
                cv2.imwrite(os.path.join(args['path_t_horizontal'], video_name), stacked_frames[:, :, j])

                if args['verbose']:
                    print('File created: %s' % os.path.join(args['path_t_horizontal'], video_name))

                # Saves vertical flow
                cv2.imwrite(os.path.join(args['path_t_vertical'], video_name), stacked_frames[:, :, j + args['nb_range']])

                if args['verbose']:
                    print('File created: %s' % os.path.join(args['path_t_vertical'], video_name))

        count += 1


def on_class_load(videos, path, class_name):
    # For each class, verifies if its folder is created in dst
    if args['spatial']:
        fm.create_dir(os.path.join(args['path_spatial'], class_name))

    if args['temporal']:
        fm.create_dirs([
            os.path.join(args['path_t_horizontal'], class_name),
            os.path.join(args['path_t_vertical'], class_name)
        ], args['verbose'])


def extract_from_dataset(src, dst, nb_frames, chunks, spatial=True, temporal=True, verbose=False):

    # Sets arguments for the script
    args['nb_frames'] = nb_frames
    args['nb_range'] = int(nb_frames / 2)
    args['chunks'] = int(chunks)
    args['spatial'] = spatial
    args['temporal'] = temporal
    args['verbose'] = verbose

    # Checks if spatial directories exist and create them
    if spatial:
        args['path_spatial'] = os.path.join(dst, 'spatial')
        fm.create_dir(args['path_spatial'], verbose)

    # Checks if spatial directories exist and create them
    if temporal:

        args['path_temporal'] = os.path.join(dst, 'temporal')
        args['path_t_horizontal'] = os.path.join(args['path_temporal'], 'u')
        args['path_t_vertical'] = os.path.join(args['path_temporal'], 'v')
        fm.create_dirs([args['path_temporal'], args['path_t_horizontal'], args['path_t_vertical']], verbose)

    # Iterates through all classes in the database files
    fm.class_iterator(path=src,
                      on_class_loaded=on_class_load,
                      process_sample=extract_video,
                      on_class_finished=None,
                      extension='.avi',
                      verbose=verbose)

    print('\nSamples extraction was finished with success!!')
