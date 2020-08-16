import keras
import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
import glob
from skimage.io import imread
from skimage.transform import resize


class MotionFlowDataGenerator(keras.utils.Sequence):

    def __init__(self, path, nb_frames, batch_size, input_shape, split='train', shuffle=True):
        # Setup dataset variables
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nb_frames = nb_frames
        self.input_shape = input_shape

        # Get classes in training folder
        self.classes = self.__get_classes()

        # Sets dataframe for sample training
        self.df = pd.read_csv(os.path.join(path, 'dataset_info.csv'))
        self.n_samples = len(self.df)
        self.indexes = self.df.index.tolist()

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

        # Prints user about those definitions
        print('Found %d classes' % len(classes))

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
        result = [self.__get_data(i) for i in indexes]

        # transform result in batch like ndarray and makes y hot encoded
        x = np.array([r[0] for r in result])
        y = to_categorical([r[1] for r in result], num_classes=len(self.classes))

        return x, y

    def __get_data(self, index):

        index = self.indexes[index]

        # Gets all sample definitions
        class_name = self.df['class'][index]
        split = self.df['split'][index]
        video_name = self.df['sample'][index]

        # Get horizontal and vertical flow path
        u_path = os.path.join(self.path, split, 'temporal', 'u', class_name)
        v_path = os.path.join(self.path, split, 'temporal', 'v', class_name)

        # Create stacked flow image
        image = np.empty(self.input_shape + (2 * self.nb_frames,))
        channel_count = 0
        for i in range(0, self.nb_frames):
            u_img = None
            v_img = None

            # Get horizontal and vertical frames
            u_img = resize(imread(os.path.join(u_path, video_name + '_u%s.jpg' % str(i).zfill(3))), self.input_shape)
            u_img = u_img - np.mean(u_img)

            v_img = resize(imread(os.path.join(v_path, video_name + '_v%s.jpg' % str(i).zfill(3))), self.input_shape)
            # v_img = np.swapaxes(v_img, 0, 1)
            v_img = v_img - np.mean(v_img)

            # Stack frames
            image[:, :, channel_count] = u_img[:, :]
            channel_count += 1
            image[:, :, channel_count] = v_img[:, :]
            channel_count += 1

        return image, self.classes.index(class_name)


if __name__ == '__main__':
    generator = MotionFlowDataGenerator(
        path=r'D:\Mestrado\databases\UCF101\test',
        split='train',
        nb_frames=10,
        batch_size=32,
        input_shape=(224, 224),
    )

    # test model
    model = keras.Sequential()
    model.add(keras.applications.VGG16(include_top=False, input_shape=(224, 224, 20), weights=None))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(3, activation='softmax', name='softmax'))

    model.summary()

    OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        OPTIMIZER,
        keras.losses.CategoricalCrossentropy(),
        metrics=['acc'],
    )

    history = model.fit_generator(
        generator,
        verbose=1,
        epochs=100,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20
    )
