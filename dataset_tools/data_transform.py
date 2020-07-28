import os
import cv2
import glob
import numpy as np

TVL1 = cv2.DualTVL1OpticalFlow_create()


def calculate_flow(frame1, frame2, bound=15):
    # Get motion flow
    flow = TVL1.calc(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)

    # Transform data back to image format
    assert flow.dtype == np.float32
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False


def extract_motion_from_frames(frames, init_frame, end_frame, nb_frames):
    motion_frames = [calculate_flow(frames[i], frames[i+1]) for i in range(init_frame, end_frame)]

    # Stacked array has shape of width * height * 2 * nb_frames
    stacked_flow = np.zeros(motion_frames[0][:, :, 1].shape + (2 * nb_frames,), dtype=np.uint8)

    for i in range(0, nb_frames):

        # Stack horizontal and vertical motion channels
        stacked_flow[:, :, i] = motion_frames[i][:, :, 0]
        stacked_flow[:, :, i + nb_frames] = motion_frames[i][:, :, 1]

    return stacked_flow


def extract_from_dataset(path, extract_path, nb_frames, chunks, spatial=True, temporal=True, verbose=False):

    # Get video classes
    classes = glob.glob(os.path.join(path, '*'))

    # Checks if directories exist and create them
    path_spatial = os.path.join(extract_path, 'spatial')
    path_temporal = os.path.join(extract_path, 'temporal')

    if create_dir(path_spatial) and verbose:
        print('Directory %s created' % path_spatial)

    if not create_dir(path_temporal) and verbose:
        print('Directory %s created' % path_temporal)

    if verbose:
        print('%d classes detected' % len(classes))

    # Extract frames for each class
    for c in classes:

        # Gets videos inside class folder and extract frames for all videos
        videos = glob.glob(os.path.join(c, '*.avi'))
        nb_range = int(nb_frames/2)

        class_name = c.split('/')[-1]
        if verbose:
            print('\nClass %s\n==================\n' % class_name)

        # Verifies if it is needed to create classes' folders
        if create_dir(os.path.join(path_spatial, class_name)) and verbose:
            print('Directory created: %s' % os.path.join(path_spatial, class_name))

        if create_dir(os.path.join(path_temporal, class_name)) and verbose:
            print('Directory created: %s' % os.path.join(path_temporal, class_name))

        # Load all frames
        for v_path in videos:

            # Extract video frames
            video = cv2.VideoCapture(v_path)
            frames = []
            success = 1
            while success:
                success, frame = video.read()

                # Appends frame only if it is not empty
                if frame is not None:
                    frames.append(frame)

            if verbose:
                print('\nVideo loaded(%d, %d, %d): %s' % (len(frames), len(frames[0]), len(frames[0][0]), v_path))

            # Finds T equally distributed indexes to represent the T video chunks
            step = float((len(frames) - 1) / (chunks + 1))
            indexes = [round(step * i) for i in range(1, chunks + 1)]

            count = 0
            for i in indexes:

                suffix = str(count).zfill(3)
                split_path = v_path.split('/')
                video_rpath = os.path.join(split_path[-2], split_path[-1])

                # get spatial frames
                if spatial:
                    file_name = video_rpath[:-4] + '_%s.jpg' % suffix
                    name = os.path.join(path_spatial, file_name)
                    cv2.imwrite(name, frames[i])
                    if verbose:
                        print('File created: %s' % name)

                # Gets optical flow frames
                if temporal:
                    file_name = video_rpath[:-4] + '_%s.npy' % suffix
                    name = os.path.join(path_temporal, file_name)
                    stacked_frames = extract_motion_from_frames(frames, i-nb_range, i+nb_range, nb_frames)
                    np.save(name, stacked_frames)
                    if verbose:
                        print('File created: %s' % name)

                count += 1

    print('\nSamples extraction was finished with success!!')


extract_from_dataset(path='/Users/ericdebaeregrassl/Documents/mestrado/datasets/test',
                     extract_path='/Users/ericdebaeregrassl/Documents/mestrado/datasets/hmdb-test',
                     nb_frames=10,
                     chunks=5,
                     verbose=True)
