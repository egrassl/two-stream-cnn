from ts_cnn import *
import keras_video.utils
import os
import glob
import keras

keras.backend.floatx()


# Custom video frame dataset loader from metal3d



# Using subdirectories path as classes
classes = [i.split(os.path.sep)[-1] for i in glob.glob('videos/*')]
classes.sort()

# Pattern to get videos and classes
glob_patterns = 'videos/{classname}/*.avi'


# Uses Keras ImageDataGenerator for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

# Create video frame generator
train = VideoFrameGenerator(
    classes=classes,
    glob_patterns=glob_patterns,
    nb_frames=10,
    split=.33,
    shuffle=True,
    batch_size=2,
    target_shape=(224, 224),
    nb_channel=3,
    transformation=data_aug
)

validation = train.get_validation_generator()
model = ts_cnn.models.TsCnnVgg16.get_stream_cnn(n_classes=len(classes), n_frames=10)
model.summary()

optimizer = ts_cnn.models.TsCnnVgg16.get_optmizer()

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

ts_cnn.models.TsCnnVgg16.train_spatial_stream("spatial_test", model, train, validation, continue_training=False)
