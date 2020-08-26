import keras
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from keras_extensions import CustomDataGenerator


class MotionFlowDataGenerator(CustomDataGenerator.CustomDataGenerator):

    def __load_flow(self, img_path, transform):
        img = imread(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.data_augmentation.apply_transform(img, transform) if transform is not None else img
        img = resize(img, self.input_shape)
        return img * (1.0/255.0)

    def get_data(self, s_path, u_path, v_path, sample_name, transform):
        # Create stacked flow image
        image = np.empty(self.input_shape + (2 * self.nb_frames,))
        channel_count = 0
        for i in range(0, self.nb_frames):
            u_img = None
            v_img = None

            # Get horizontal and vertical frames
            u_img_path = os.path.join(u_path, sample_name + '_u%s.jpg' % str(i).zfill(3))
            u_img = self.__load_flow(u_img_path, transform)

            v_img_path = os.path.join(v_path, sample_name + '_v%s.jpg' % str(i).zfill(3))
            v_img = self.__load_flow(v_img_path, transform)

            # Stack frames
            image[:, :, channel_count] = u_img[:, :, 0]
            channel_count += 1
            image[:, :, channel_count] = v_img[:, :, 0]
            channel_count += 1

        return image


if __name__ == '__main__':

    # Creates data Augmentation generator
    image_aug = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.3,
        horizontal_flip=True,
        rotation_range=60,
        width_shift_range=.25,
        height_shift_range=.25,
        channel_shift_range=.35,
        brightness_range=[.3, 1.5]
    )

    generator = MotionFlowDataGenerator(
        path=r'D:\Mestrado\databases\UCF101\test',
        split='train',
        nb_frames=10,
        batch_size=4,
        input_shape=(224, 224),
        data_augmentation=image_aug
    )

    val_generator = MotionFlowDataGenerator(
        path=r'D:\Mestrado\databases\UCF101\test',
        split='val',
        nb_frames=10,
        batch_size=4,
        input_shape=(224, 224)
        # data_augmentation=image_aug
    )

    # test model
    model = keras.Sequential()
    model.add(keras.applications.VGG16(include_top=False, input_shape=(224, 224, 20), weights=None))
    td = keras.layers.TimeDistributed(input_shape=(5, 224, 224, 20), layer=model)
    model = keras.Sequential()
    model.add(td)
    model.add(keras.layers.GlobalMaxPooling3D())
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
        validation_data=val_generator,
        verbose=1,
        epochs=100,
        use_multiprocessing=True,
        workers=8
        # max_queue_size=20
    )
