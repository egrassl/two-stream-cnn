import argparse
import os
import glob
import utils.file_management as fm

# Creates checkup directory if it does not exists
fm.create_dir(os.path.join(os.getcwd(), 'chkp'))

# ====== Argument Parser ======
parser = argparse.ArgumentParser()

# Obligatory arguments
parser.add_argument('n', metavar='model_name', type=str, help='Name given to the cnn model. It is used in log files')
parser.add_argument('t', metavar='stream_type', type=str, choices=['s', 't', 'st'], help='Chooses which stream will be used: spatial, temporal or both')
parser.add_argument('dataset', metavar='dataset_path', type=str, help='Path to videos dataset')
parser.add_argument('-e', metavar='epochs', type=int, help='How many epoch to train the network', required=True)
parser.add_argument('--val', metavar='Motion frames', type=str, help='Number of motion frames per sample', required=True)

# Optional arguments
parser.add_argument('--weights', metavar='weights_file_path or \'imagenet\'', type=str, default=None, help="File path to load previously trained weights")
parser.add_argument('--new', default=False, action='store_true', help="Indicates to overwrite logfile if it exists")
parser.add_argument('--init', metavar='initial_epoch', type=int, default=1, help='Used to continue from a previous training')
parser.add_argument('--gpu', metavar='gpu_id', type=str, help='Specifies on which GPU to run')
parser.add_argument('--bs', metavar='batch_size', type=int, default=8, help='Batch size using during training')

args = parser.parse_args()

# ===== process arguments ======

# Specifies a GPU to use
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Get classes by dataset's subdirectories
classes = [i.split(os.path.sep)[-1] for i in glob.glob(os.path.join(args.dataset, '*'))]


# Keras related packages are imported after GPU argument to respect the GPU choice
import ts_cnn.models as ts
import keras
import keras_extensions.preprocess_crop
import keras_extensions.callbacks as callbacks
import keras_extensions.MotionFlowDataGenerator as mfd
import keras_extensions.SpatialDataGenerator as mfs

# ========== Training parameters ==========
LEARNING_RATE = 1e-5 # np.power(50., -4.)
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# OPTIMIZER = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=.9)
DROPOUT = .85
L2 = 5e-4
# N_CLASSES = len(classes)
N_CLASSES = 101
FC_LAYERS = 3
FC_NEURONS = 4096
MODEL = ts.CNNType.VGG16
NB_FRAMES = 10

# Callback parameters
DECAY_PATIENCE = 8
E_STOP_PATIENCE = 15


def l_schedule(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * .1


# Keras callbacks
callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=DECAY_PATIENCE, verbose=1),
        # keras.callbacks.LearningRateScheduler(l_schedule, verbose=True),
        keras.callbacks.EarlyStopping(patience=E_STOP_PATIENCE),
        callbacks.CustomCheckpointCallback(filename=os.path.join('chkp/', '%s_best.h5' % args.n), save_best_only=True),
        callbacks.CustomCheckpointCallback(filename=os.path.join('chkp/', '%s_last.h5' % args.n), save_best_only=False),
        keras.callbacks.CSVLogger(filename='chkp/%s.hist' % args.n, separator=',', append=not args.new)
    ]

if MODEL == ts.CNNType.XCEPTION:
    INPUT_SIZE = (299, 299)
else:
    INPUT_SIZE = (224, 224)

# Creates data Augmentation generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.3,
        horizontal_flip=True,
        rotation_range=25,
        width_shift_range=.25,
        height_shift_range=.25,
        channel_shift_range=.35,
        brightness_range=[.5, 1.5],
        rescale=1.0/255.0
    )

# Uses default image flow from Keras if it is a spatial stream
if args.t == 's':
    # Train
    train_set = mfs.SpatialDataGenerator(
        path=args.dataset,
        split='train',
        nb_frames=NB_FRAMES,
        batch_size=args.bs,
        input_shape=INPUT_SIZE,
        data_augmentation=train_datagen
    )

    val_set = mfs.SpatialDataGenerator(
        path=args.dataset,
        split='val',
        nb_frames=NB_FRAMES,
        batch_size=args.bs,
        input_shape=INPUT_SIZE,
        # data_augmentation=train_datagen
        # data_augmentation=image_aug
    )

    # Validation
    #val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    #val_set = val_datagen.flow_from_directory(args.val, target_size=INPUT_SIZE, color_mode='rgb')

elif args.t == 't':
    train_set = mfd.MotionFlowDataGenerator(
        path=args.dataset,
        split='train',
        nb_frames=NB_FRAMES,
        batch_size=args.bs,
        input_shape=INPUT_SIZE
        # data_augmentation=image_aug
    )

    val_set = mfd.MotionFlowDataGenerator(
        path=args.dataset,
        split='val',
        nb_frames=NB_FRAMES,
        batch_size=args.bs,
        input_shape=INPUT_SIZE
        # data_augmentation=image_aug
    )

else:
    raise NotImplemented()


model = ts.TSCNN(
    cnn_model=MODEL,
    stream_type=args.t,
    n_classes=N_CLASSES,
    fc_layers=FC_LAYERS,
    fc_neurons=FC_NEURONS,
    nb_frames=NB_FRAMES,
    dropout=DROPOUT,
    l2=L2,
    chunks=5,
    temporal_weights='a',
    spatial_weights='imagenet',
    st_weights='a'
    #callbacks=callbacks,
    #weights=args.weights
)

model.model.summary()

answer = input('Do you want to continue with the model printed above? (yes|no): ')
if answer != 'yes':
    exit(0)

# Compiles and train the model
model.model.compile(
    OPTIMIZER,
    keras.losses.CategoricalCrossentropy(),
    metrics=['acc']
)

history = model.model.fit_generator(
    train_set,
    validation_data=val_set,
    verbose=1,
    epochs=args.e,
    initial_epoch=args.init - 1,
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=8
)
