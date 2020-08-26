import os
import cv2
import utils.file_management as fm
import utils.video_processing as vp
import pandas as pd


class DataExtract(object):

    def __init__(self, src, dst, nb_frames, chunks, split_func, spatial, temporal, verbose):
        self.src = src
        self.dst = dst
        self.nb_frames = nb_frames
        self.nb_range = int(nb_frames / 2)
        self.chunks = chunks
        self.spatial = spatial
        self.temporal = temporal
        self.verbose = verbose
        self.split_func = split_func

        # Setup CSV classes file
        self.data_frame = pd.DataFrame(columns=['class', 'split', 'sample', 'nb_chunks'])
        self.csv_file = os.path.join(dst, 'dataset_info.csv')
        self.sample_count = 0

        # set splits directories and creates them if needed
        self.train_dst = os.path.join(dst, 'train')
        self.val_dst = os.path.join(dst, 'val')
        self.test_dst = os.path.join(dst, 'test')

        split_dirs = [self.train_dst, self.val_dst, self.test_dst]
        fm.create_dirs(split_dirs, self.verbose)

        # Create data directories for each split directory if needed
        for split in split_dirs:
            s_dst = os.path.join(split, 'spatial')
            u_dst = os.path.join(split, 'temporal', 'u')
            v_dst = os.path.join(split, 'temporal', 'v')

            if spatial:
                fm.create_dir(s_dst, self.verbose)

            if temporal:
                fm.create_dirs([os.path.join(split, 'temporal'), u_dst, v_dst], self.verbose)

    def on_class_load(self, videos, path, class_name):
        '''
        Initializes the destiny directory for a class on all splits directories
        '''

        split_dirs = [self.train_dst, self.val_dst, self.test_dst]
        # Create data directories for each class directory in each split if needed
        for split in split_dirs:
            s_dst = os.path.join(split, 'spatial', class_name)
            u_dst = os.path.join(split, 'temporal', 'u', class_name)
            v_dst = os.path.join(split, 'temporal', 'v', class_name)

            if self.spatial:
                fm.create_dir(s_dst, self.verbose)

            if self.temporal:
                fm.create_dirs([u_dst, v_dst], self.verbose)

    def get_indexes(self, n_frames, offset=.2):
        '''
        Gets the T distributed indexes over the video frames. It returns less indexes if could not get self.chunks indexes

        :param n_frames: number of frames in the video
        :param offset: offset on the left of the video (to avoid the starting frames, for example)
        :return: a list of frame indexes and its quantity
        '''
        offset = int(n_frames * offset)
        n = n_frames - offset

        # Try to find T chunks, but a smaller chunk quantity will be extracted if video is too short
        i = 0
        indexes = [n]
        while indexes[-1] + self.nb_range >= n:
            step = float((n - 1) / (self.chunks + 1 - i))
            indexes = [round(step * i) for i in range(1, self.chunks + 1 - i)]
            i += 1

        if self.verbose:
            print('%d chunks will be extracted' % len(indexes))

        return [i + offset for i in indexes]

    def get_video_dst(self, video_name):
        '''
        Defines if sample is a training, validation or test sample bases on the split_func property

        :param video_name: video name or path that will be used as parameter to split_func
        :return: spatial, horizontal flow and vertical flow destiny paths
        '''
        #
        split = self.split_func(video_name)
        if split == 'train':
            dst = self.train_dst
        elif split == 'val':
            dst = self.val_dst
        else:
            dst = self.test_dst

        s_dst = os.path.join(dst, 'spatial')
        u_dst = os.path.join(dst, 'temporal', 'u')
        v_dst = os.path.join(dst, 'temporal', 'v')

        return s_dst, u_dst, v_dst, split

    def process_video(self, frames, video_name, class_path, class_name):

        # Defines if sample is a training, validation or test sample
        s_dst, u_dst, v_dst, split = self.get_video_dst(video_name)

        if self.verbose:
            print('\n(%d, %d, %d) video loaded: %s' % (len(frames), len(frames[0]), len(frames[0][1]), video_name))

        # Get index for each central frame of each chunk
        indexes = self.get_indexes(len(frames))
        self.sample_count += 1

        # Adds sample definitions to csv
        self.data_frame.loc[self.sample_count] = [class_name, split, video_name[:-4], len(indexes)]

        count = 0
        for i in indexes:

            suffix = str(count).zfill(3)

            # Adds sample definitions to csv
            # self.data_frame.loc[self.sample_count] = [class_name, split, video_name[:-4] + '_%s' % suffix, len(indexes)]
            # self.sample_count += 1

            # Saves spatial frame if it doesn't exist yet
            if self.spatial:

                file_path = os.path.join(s_dst, class_name, video_name)[:-4] + '_%s.jpg' % suffix

                if os.path.isfile(file_path):
                    print('File already exists, thus extraction was skipped: %s' % file_path)

                else:
                    cv2.imwrite(file_path, vp.frame_resize(frames[i]))

                    if self.verbose:
                        print('File created: %s' % file_path)

            # Saves temporal frames if they don't exist yet
            if self.temporal:
                index_count = 0
                for j in range(i - self.nb_range, i + self.nb_range):
                    u_file_path = os.path.join(u_dst, class_name, video_name)[:-4] + '_%s_u%s.jpg' % (suffix, str(index_count).zfill(3))
                    v_file_path = os.path.join(v_dst, class_name, video_name)[:-4] + '_%s_v%s.jpg' % (suffix, str(index_count).zfill(3))

                    # Calculates motion flow only if file does not exists
                    if os.path.isfile(u_file_path) and os.path.isfile(v_file_path):
                        print('File already exists, thus extraction was skipped: %s' % u_file_path)
                        print('File already exists, thus extraction was skipped: %s' % v_file_path)

                    else:
                        # Get motion flow
                        flow = vp.calculate_flow(vp.frame_resize(frames[j]), vp.frame_resize(frames[j+1]), flow_type=1)

                        cv2.imwrite(u_file_path, flow[:, :, 0])
                        cv2.imwrite(v_file_path, flow[:, :, 1])

                        if self.verbose:
                            print('File created: %s' % u_file_path)
                            print('File created: %s' % v_file_path)

                    index_count += 1

            count += 1

    def extract(self):
        '''
        Iterates though all classes of the src dataset and extract the dataset to dst
        :return:
        '''
        # Iterates through all classes
        fm.class_iterator(path=self.src,
                          on_class_loaded=self.on_class_load,
                          process_sample=self.process_video,
                          on_class_finished=None,
                          extension='.avi',
                          verbose=self.verbose)

        # Writes dataset csv file
        self.data_frame.to_csv(self.csv_file)

        if self.verbose:
            print('\nSamples extraction was finished with success!!')
