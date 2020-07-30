import os
import keras
import utils.file_management as fm
import numpy as np

DROPOUT = .85

def l_schedule(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * 0.1

def add_fully_connected(model, n_classes):

    # model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.GlobalAveragePooling2D())

    # Add fully Connected Layers
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'))

    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'))

    # Dense layer + softmax layer
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))


def xception_spatial(n_classes, weights):
    model = keras.Sequential()

    # Creates Xception model for image classification
    model.add(keras.applications.Xception(input_shape=(299, 299, 3), weights=weights, include_top=False))

    for layer in model.layers: layer.W_regularizer = keras.regularizers.l2(1e-5)

    # Adds classification network for specified classes
    add_fully_connected(model, n_classes)
    return model


def xception_temporal(n_classes, nb_frames):
    model = keras.Sequential()

    # Creates Xception model for image classification
    model.add(keras.applications.Xception(input_shape=(299, 299, 2 * nb_frames), weights=None, include_top=False))

    # Adds classification network for specified classes
    add_fully_connected(model, n_classes)
    return model


def train_stream(name, model, train, validation, initial_epoch, weights_path, epochs, new=False):

    # Create chkp dir if needed
    if fm.create_dir('chkp'):
        print('Directory created: %s' % os.path.join(os.getcwd(), 'chkp'))

    # Checks if it can load weights
    if weights_path is not None:

        if os.path.isfile(weights_path):
            model.load_weights(weights_path)

        else:
            raise Exception('Weights file path does not exists: %s' % weights_path)

    # Training parameters
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=10, min_lr=1e-6),
        keras.callbacks.LearningRateScheduler(l_schedule, verbose=True),
        keras.callbacks.ModelCheckpoint(
            'chkp/' + name + '.hdf5',
            save_weights_only=True,
            save_best_only=True,
            verbose=1),
        keras.callbacks.CSVLogger(filename='chkp/%s.hist' % name, separator=',', append=not new)
    ]

    # Train generator
    history = model.fit_generator(
        train,
        validation_data=validation,
        verbose=1,
        epochs=epochs,
        initial_epoch=initial_epoch - 1,
        callbacks=callbacks
     )

    return history
