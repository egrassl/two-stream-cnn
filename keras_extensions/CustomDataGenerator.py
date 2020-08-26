import keras
from keras.utils import to_categorical
import pandas as pd
import os
import glob
import numpy as np


class CustomDataGenerator(keras.utils.Sequence):

    def __init__(self, path, nb_frames, batch_size, input_shape, split='train', shuffle=True,
                 data_augmentation: keras.preprocessing.image.ImageDataGenerator = None):
        # Setup dataset variables
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nb_frames = nb_frames
        self.input_shape = input_shape
        self.data_augmentation = data_augmentation
        self.split = split

        # Get classes in training folder
        self.classes = self.__get_classes()

        # Sets dataframe for sample training
        self.df = pd.read_csv(os.path.join(path, 'dataset_info.csv'))

        # Drop df rows that are not from the specified split
        split_indexes = []
        for i in range(0, len(self.df)):
            if self.df['split'][i] != self.split:
                split_indexes.append(i)
        self.df = self.df.drop(self.df.index[split_indexes])
        print('Found %d classes and %d images for %s split!!' % (len(self.classes), len(self.df), self.split))

        self.indexes = self.df.index.tolist()
        self.n_samples = len(self.df)

        self.on_epoch_end()

    def __get_classes(self):
        '''
        Returns an array with all classes name in the training folder of the dataset
        '''

        # Finds how many classes folders there are in the train dataset
        u_path = os.path.join(self.path, 'train', 'temporal', 'u', '*')
        classes = glob.glob(os.path.join(u_path))
        classes = [os.path.split(c)[1] for c in classes]
        classes.sort()

        return classes

    def __len__(self):
        '''
        Returns the total of batches for the training
        '''
        return self.n_samples // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        '''
        Gets the batch for training
        '''

        # Gets samples indexes
        indexes = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]
        result = [self.__get_sample(i) for i in indexes]

        # transform result in batch like ndarray and makes y hot encoded
        x = np.array([r[0] for r in result])
        y = to_categorical([r[1] for r in result], num_classes=len(self.classes))

        return x, y

    def __get_sample(self, index):

        index = self.indexes[index]

        # Gets all sample definitions
        class_name = self.df['class'][index]
        split = self.df['split'][index]
        video_name = self.df['sample'][index]
        chunks = self.df['nb_chunks'][index]

        # Get spatial path
        s_path = os.path.join(self.path, split, 'spatial', class_name)

        # Get horizontal and vertical flow path
        u_path = os.path.join(self.path, split, 'temporal', 'u', class_name)
        v_path = os.path.join(self.path, split, 'temporal', 'v', class_name)

        # Create stacked flow image
        shape = (chunks,) + self.input_shape + (2 * self.nb_frames,)
        images = np.empty(shape)
        transform = self.data_augmentation.get_random_transform(self.input_shape) if self.data_augmentation is not None else None

        for i in range(0, chunks):
            sample_name = video_name + '_%s' % str(i).zfill(3)
            images[i] = self.get_data(s_path, u_path, v_path, sample_name, transform)

        return images, self.classes.index(class_name)

    def get_data(self, s_path, u_path, v_path, sample_name, transform):

        raise Exception('The base class must implement __get_data method!!')
