import os
import keras
import utils.file_management as fm
import keras_extensions.model_funcs as mf
from enum import Enum


class CNNType(Enum):
    VGG16 = 1
    XCEPTION = 2


class TSCNN(object):

    def __init__(self, cnn_model, stream_type, n_classes, fc_layers, fc_neuros, nb_frames, dropout, l2, callbacks, weights, spatial_weights, temporal_weights):
        self.cnn_model = cnn_model
        self.stream_type = stream_type
        self.n_classes = n_classes
        self.fc_layers = fc_layers
        self.fc_neurons = fc_neuros
        self.nb_frames = nb_frames
        self.dropout = dropout
        self.l2 = l2
        self.callbacks = callbacks
        self.weights = weights
        self.spatial_weights = spatial_weights
        self.temporal_weights = temporal_weights

        self.l2_reg = keras.regularizers.l2(l2)

        self.model = self.get_model()

    def add_l2_regularizer(self, model):
        '''
        Adds L2 regularizer for each model Layer

        :param model: model to be regularized
        :return: new model object with regularization
        '''
        # Adds regularizer to all layers
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, self.l2_reg)

        # Keras bug fix to update the model after regularization
        # Checks if tmp folder exists
        fm.create_dir(os.path.join(os.getcwd(), 'tmp'))
        tmp_weights_path = os.path.join(os.getcwd(), 'tmp', 'tmp_weights.h5')

        if os.path.isfile(tmp_weights_path):
            os.remove(tmp_weights_path)

        mf.save_model(model, tmp_weights_path)

        model = mf.reload_model(model)

        mf.load_weights(model, tmp_weights_path)

        # Remove old weights file
        os.remove(tmp_weights_path)

        return model

    def get_cnn(self, stream_type, name='cnn'):
        '''
        Returns the cnn model for the given stream

        :param stream_type: 's' for spatial and 't' for temporal
        :return: a keras cnn model
        '''

        if stream_type != 's' and stream_type != 't':
            raise Exception('Stream type can only be s for spatial and t for temporal')

        cnn_switch = {
            CNNType.VGG16: (keras.applications.VGG16, (224, 224, 3) if stream_type == 's' else (224, 224, 2 * self.nb_frames)),
            CNNType.XCEPTION: (keras.applications.Xception, (299, 299, 3) if stream_type == 's' else (299, 299, 2 * self.nb_frames)),
        }

        # Check if imagenet pre trained model will be used
        if self.weights == 'imagenet':
            print('Using imagenet Weights...')
            weights = 'imagenet'
        else:
            weights = None

        input_shape = cnn_switch[self.cnn_model][1]

        cnn = cnn_switch[self.cnn_model][0](input_shape=input_shape, weights=weights, include_top=False)

        if self.l2 is not None:
            print('Adding L2 regularization...')
            cnn = self.add_l2_regularizer(cnn)

        # Returns Cnn with batch normalization on input

        cnn.name = name
        return keras.Sequential([
            keras.layers.BatchNormalization(input_shape=input_shape),
            cnn
        ])

    def add_layer_regularized(self, layer, n_neurons, activation, model):
        '''
        Adds a dense layer with L2 if specified

        :param layer: Keras layer function
        :param n_neurons: Number of neurons in the layer
        :param activation: Activation function
        :param model: Model to add layer
        '''
        if self.l2 is not None:
            model.add(layer(n_neurons, activation=activation, kernel_regularizer=self.l2_reg))
        else:
            model.add(layer(n_neurons, activation=activation))

    def add_fully_connected(self, model):
        '''
        Adds fully connected layers to the model (up to softmax layer)

        :param model: model to add layers
        '''
        model.add(keras.layers.Flatten())

        # Adds FC layers excel softmax
        for i in range(0, self.fc_layers - 1):
            # Adds regularization if specified
            self.add_layer_regularized(keras.layers.Dense, self.fc_neurons, 'relu', model)

            # Adds dropout if specified
            if self.dropout > 0:
                model.add(keras.layers.Dropout(self.dropout))

        # Adds softmax layers
        self.add_layer_regularized(keras.layers.Dense, self.n_classes, 'softmax', model)

    def get_model(self):
        model = keras.Sequential()

        if self.stream_type == 's':
            model.add(self.get_cnn('s'))
        elif self.stream_type == 't':
            model.add(self.get_cnn('t'))
        else:

            # Checks if both streams weighs were provided
            if self.spatial_weights is None or not os.path.isfile(self.spatial_weights) or self.temporal_weights is None or not os.path.isfile(self.temporal_weights):
                raise Exception('both spatial and temporal weights files must be provided to train this model!')

            spatial = self.get_cnn('s', 'cnn_spatial')
            mf.load_weights(spatial, self.spatial_weights)
            for layer in spatial.layers:
                layer.trainable = False

            temporal = self.get_cnn('t', 'cnn_temporal')
            mf.load_weights(temporal, self.temporal_weights)
            for layer in temporal.layers:
                layer.trainable = False

            combined = keras.layers.concatenate([spatial.output, temporal.output])

            # fc = keras.layers.Conv3D(filters=(1, 1, 1024), kernel_size=(3, 3, 3))(combined)
            # fc = keras.layers.GlobalMaxPooling3D()(fc)
            fc = keras.layers.Dense(4096, activation='relu', kernel_regularizer=self.l2_reg)(combined)
            fc = keras.layers.Dense(4096, activation='relu', kernel_regularizer=self.l2_reg)(fc)
            fc = keras.layers.Dense(self.n_classes, activation='softmax', kernel_regularizer=self.l2_reg)(fc)

            model = keras.Model(inputs=[spatial.input, temporal.input], outputs=fc)

            return model

        # Adds FC layers
        self.add_fully_connected(model)

        # loads weights if it is a file
        if self.weights is not None and os.path.isfile(self.weights):
            mf.load_weights(model, self.weights)
            print('Weights loaded from %s' % self.weights)

        return model


if __name__ == '__main__':

    model = TSCNN(
        cnn_model=CNNType.VGG16,
        stream_type='st',
        n_classes=3,
        fc_layers=3,
        fc_neuros=4096,
        nb_frames=10,
        dropout=.85,
        l2=1e-5,
        callbacks=[],
        weights=None,
        spatial_weights=r'/home/coala/mestrado/ts-cnn/chkp/joint_test_spatial_best.h5',
        temporal_weights=r'/home/coala/mestrado/ts-cnn/chkp/joint_test_temporal_best.h5'
    )

    OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-5)

    # Compiles and train the model
    model.model.compile(
        OPTIMIZER,
        keras.losses.CategoricalCrossentropy(),
        metrics=['acc']
    )

    model.model.summary()

