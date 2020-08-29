import keras
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from keras_extensions import CustomDataGenerator


class SpatialDataGenerator(CustomDataGenerator.CustomDataGenerator):

    def get_data(self, s_path, u_path, v_path, sample_name, transform):
        s_path = os.path.join(s_path, sample_name + '.jpg')
        img = imread(s_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.data_augmentation.apply_transform(img, transform) if transform is not None else img
        img = resize(img, self.input_shape)
        return img * (1.0 / 255.0)


if __name__ == '__main__':

    # Creates data Augmentation generator
    image_aug = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.3,
        horizontal_flip=True,
        rotation_range=25,
        width_shift_range=.25,
        height_shift_range=.25,
        channel_shift_range=.35,
        brightness_range=[.5, 1.5]
    )

    generator = SpatialDataGenerator(
        path=r'/home/coala/mestrado/datasets/UCF003/',
        split='train',
        nb_frames=10,
        batch_size=4,
        input_shape=(224, 224),
        data_augmentation=image_aug
    )

    val_generator = SpatialDataGenerator(
        path=r'/home/coala/mestrado/datasets/UCF003/',
        split='val',
        nb_frames=10,
        batch_size=4,
        input_shape=(224, 224)
        # data_augmentation=image_aug
    )

    # test model
    model = keras.Sequential()
    model.add(keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))
    td = keras.layers.TimeDistributed(input_shape=(5, 224, 224, 3), layer=model)
    model = keras.Sequential()
    model.add(td)
    # model.add(keras.layers.Conv3D(filters=256, kernel_size=(2, 2, 2), strides=2))
    model.add(keras.layers.MaxPool3D(pool_size=(5, 5, 5)))
    # model.add(keras.layers.GlobalAvgPool3D())
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(3, activation='softmax', name='softmax'))

    model.summary()

    OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-5)

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
        #use_multiprocessing=True,
        #workers=8
    )
