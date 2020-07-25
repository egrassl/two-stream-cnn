import argparse
import os
import glob

# ====== Argument Parser ======

parser = argparse.ArgumentParser()

# Obligatory arguments
parser.add_argument('-n', metavar='model_name', type=str, help='Name given to the cnn model. It is used in log files',
                    required=True)
parser.add_argument('-t', metavar='stream_type', type=str, choices=['s', 't', 'st'], required=True,
                    help='Chooses which stream will be used: spatial, temporal or both')
parser.add_argument('--dataset', metavar='dataset_path', type=str, help='Path to videos dataset', required=True)
parser.add_argument('-e', metavar='epochs', type=int, help='How many epoch to train the network', required=True)
parser.add_argument('--log', metavar='log_file_path', type=str, help='File path to keep training logs', required=True)


# Optional arguments
parser.add_argument('--new', default=False, action='store_true',
                    help="Indicates to overwrite logfile if it exists")
parser.add_argument('--load_weights', metavar='weights_file_path', type=str,
                    help="File path to load previously trained weights")
parser.add_argument('--init', metavar='initial_epoch', type=int, default=1,
                    help='Used to continue from a previous training')
parser.add_argument('--gpu', metavar='gpu_id', type=str, help='Specifies on which GPU to run')
parser.add_argument('--bs', metavar='batch_size', type=int, default=8, help='Batch size using during training')

args = parser.parse_args()

# ===== process arguments ======

# Specifies a GPU to use
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Get classes by dataset's subdirectories
classes = [i.split(os.path.sep)[-1] for i in glob.glob(os.path.join(args.dataset, '*'))]
classes.sort()

# Keras related packages are imported after GPU argument to respect the GPU choice
import keras_video
from ts_cnn.models import *

# calculates how many classes in the dataset directory

if args.t == 's':
    model = TsCnnVgg16.get_stream_cnn(len(classes), 10)

    # Uses Keras ImageDataGenerator for data augmentation
    data_aug = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2)

    print('\n====== DATASET INFO ======')
    # Create video frame generator
    train = keras_video.VideoFrameGenerator(
        classes=classes,
        glob_patterns=os.path.join(args.dataset, '{classname}/*'),
        nb_frames=10,
        split=.33,
        shuffle=True,
        batch_size=args.bs,
        target_shape=(224, 224),
        nb_channel=3,
        transformation=data_aug
    )
    print('\n')


# Gets validation dataset
validation = train.get_validation_generator()

model.summary()

# Trains the model
optimizer = TsCnnVgg16.get_optmizer()
model.compile(
    optimizer,
    #'categorical_crossentropy',
    keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc']
)

history = TsCnnVgg16.train_spatial_stream(model_name=args.n, model=model, train=train, validation=validation,
                                          epochs=args.e, history_file=args.log, initial_epoch=args.init,
                                          weights_file=args.load_weights, continue_training=not args.new)

