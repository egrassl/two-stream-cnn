import os
import keras


class TsCnnVgg16(object):

    # Adds the fully connected network in a model
    @staticmethod
    def add_fully_connected(model, n_classes):
        # Dense layer 1
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'))
        #model.add(keras.layers.Dropout(0.85))

        # Dense layer 2
        #model.add(keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'))
        #model.add(keras.layers.Dropout(0.85))

        # Dense layer 3 + softmax layer
        model.add(keras.layers.Dense(n_classes, activation='softmax'))

    @staticmethod
    def get_stream_cnn(n_classes, n_frames, weights='imagenet', input_shape=(224, 224, 3)):
        # Calculates shape over distributed time of n_frames frames
        shape = (n_frames,) + input_shape

        # Model used for CNN is VGG16 with a time distributed layer to process all input frames
        vgg16 = keras.applications.Xception(include_top=False, input_shape=input_shape, weights=weights)
        vgg_pool = keras.Sequential([vgg16, keras.layers.GlobalAveragePooling2D()])
        model = keras.Sequential(keras.layers.TimeDistributed(vgg_pool, input_shape=shape))

        # model.summary()

        # Adds classification network
        TsCnnVgg16.add_fully_connected(model, n_classes)
        return model

    @staticmethod
    def get_optmizer():
        return keras.optimizers.SGD(learning_rate=0.000008, momentum=0.9)

    @staticmethod
    def train_spatial_stream(model_name, model, train, validation, history_file, initial_epoch, weights_file,
                             epochs, continue_training):

        # Check for historical files and weights from previous trainings if continue is necessary
        #current_path = os.getcwd()


        # Check if the log file exists
        if continue_training and not os.path.isfile(history_file):
            raise Exception('Could not find the history file:' + history_file)

        if weights_file is not None:

            # check if weight file exists
            if os.path.isfile(weights_file):
                model.load_weights(weights_file)

            # Checks if weight file exists
            else:
                raise Exception('Could not find the weight file:' + weights_file)

        # Create callbacks
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=0.0000008),
            keras.callbacks.ModelCheckpoint(
                'chkp/' + model_name + '.{epoch:03d}.hdf5',
                save_weights_only=True,
                save_best_only=True,
                verbose=1),
            keras.callbacks.CSVLogger(filename=history_file, separator=',', append=continue_training)
        ]

        # Train generator
        history = model.fit_generator(
            train,
            validation_data=validation,
            verbose=1,
            epochs=epochs,
            initial_epoch=initial_epoch-1,
            callbacks=callbacks
         )

        return history
