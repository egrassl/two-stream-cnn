import os
import keras
import utils.file_management as fm


def add_fully_connected(model, n_classes):
    # Adds
    model.add(keras.layers.GlobalAveragePooling2D())

    # Dense layer + softmax layer
    model.add(keras.layers.Dense(n_classes, activation='softmax'))


def xception_spatial(n_classes, weights):
    model = keras.Sequential()

    # Creates Xception model for image classification
    model.add(keras.applications.Xception(input_shape=(299, 299, 3), weights=weights, include_top=False))

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
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0000008),
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
