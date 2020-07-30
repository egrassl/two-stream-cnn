import argparse
import os
import glob
import PIL

# ====== Argument Parser ======

parser = argparse.ArgumentParser()

# Obligatory arguments
parser.add_argument('-n', metavar='model_name', type=str, help='Name given to the cnn model. It is used in log files',
                    required=True)
parser.add_argument('-t', metavar='stream_type', type=str, choices=['s', 't', 'st'], required=True,
                    help='Chooses which stream will be used: spatial, temporal or both')
parser.add_argument('--dataset', metavar='dataset_path', type=str, help='Path to videos dataset', required=True)
parser.add_argument('-e', metavar='epochs', type=int, help='How many epoch to train the network', required=True)
parser.add_argument('--val', metavar='Motion frames', type=str, help='Number of motion frames per sample', required=True)


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
import ts_cnn.models as ts
import keras

# OPTIMIZER = keras.optimizers.SGD(learning_rate=0.000008, momentum=0.9)
OPTIMIZER = keras.optimizers.Adam(learning_rate=.0000008)

if args.t == 's':
    model = ts.xception_spatial(len(classes), 'imagenet')

    # Uses Keras ImageDataGenerator for data augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=.2,
        height_shift_range=.2,
        channel_shift_range=.2,
        rescale=1.0/255.0
    )

    train_set = train_datagen.flow_from_directory(
        args.dataset,
        target_size=(299, 299),
        batch_size=args.bs,
        subset='training',
        color_mode='rgb',
        shuffle=True
    )

    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    validation_set = val_datagen.flow_from_directory(
        args.val,
        target_size=(299, 299),
    )


# Gets validation dataset


model.summary()

# Trains the model
model.compile(
    OPTIMIZER,
    keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['acc']
)

history = ts.train_stream(args.n, model, train_set, validation_set, args.init, None, args.e)
