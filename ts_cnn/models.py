import os
import keras
import utils.file_management as fm
import keras_extensions.model_funcs as mf
from enum import Enum


class CNNType(Enum):
    VGG16 = 1
    XCEPTION = 2


class TSCNN(object):

    def __init__(self, cnn_model: CNNType, stream_type: str, n_classes: int, fc_layers: int, fc_neurons: int,
                 chunks: int, nb_frames: int, dropout: float, l2: float, st_weights: str = None,
                 spatial_weights: str = None, temporal_weights: str = None):

        # Cnn model (VGG or Xception) and stream type
        self.cnn_model = cnn_model
        self.stream_type = stream_type

        # Parameters used to build the model
        self.n_classes = n_classes
        self.fc_layers = fc_layers
        self.fc_neurons = fc_neurons
        self.nb_chunks = chunks
        self.nb_frames = nb_frames
        self.dropout = dropout
        self.l2_reg = keras.regularizers.l2(l2) if l2 is not None else None

        # Weights definitions (loaded by name)
        self.st_weights = st_weights
        self.spatial_weights = spatial_weights
        self.temporal_weights = temporal_weights

        # Build the model that will be trained
        self.model = self.get_model()

    def __add_l2_regularizer(self, model):
        '''
        Adds L2 regularizer for each model Layer

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

        # Remove saving file if it already exited
        if os.path.isfile(tmp_weights_path):
            os.remove(tmp_weights_path)

        # Reload model structure and weights
        mf.save_model(model, tmp_weights_path)
        model = mf.reload_model(model)
        mf.load_weights(model, tmp_weights_path)

        # Remove weights file
        os.remove(tmp_weights_path)

        return model

    def __get_cnn(self, stream_type: str, imagenet: bool):
        '''
        Returns the cnn model for the given stream

        :param stream_type: 's' for spatial and 't' for temporal
        :return: a keras cnn model
        '''

        if stream_type != 's' and stream_type != 't':
            raise Exception('Stream type can only be s for spatial and t for temporal')

        # Defines CNNs combinations
        cnn_switch = {
            CNNType.VGG16: (keras.applications.VGG16, (224, 224, 3) if stream_type == 's' else (224, 224, 2 * self.nb_frames)),
            CNNType.XCEPTION: (keras.applications.Xception, (299, 299, 3) if stream_type == 's' else (299, 299, 2 * self.nb_frames)),
        }

        if imagenet:
            print('Loading default CNN\'s imagenet weights')

        # Loads CNN model from keras applications
        weights = 'imagenet' if imagenet else None
        input_shape = cnn_switch[self.cnn_model][1]
        cnn = cnn_switch[self.cnn_model][0](input_shape=input_shape, weights=weights, include_top=False)

        # Adds L2 regularization if specified
        if self.l2_reg is not None:
            print('Adding L2 regularization...')
            cnn = self.__add_l2_regularizer(cnn)

        # Name the cnn for weight loading in the future
        cnn.name = 'Spatial_stream_CNN' if stream_type == 's' else 'Temporal_stream_CNN'

        # Creates a time distributed CNN and return it
        inputs = keras.layers.Input(shape=(5,) + input_shape)
        outputs = keras.layers.TimeDistributed(cnn)(inputs)

        return inputs, outputs

    def __layer_regularized(self, layer, n_neurons, activation):
        '''
        Adds a dense layer with L2 if specified

        :param layer: Keras layer function
        :param n_neurons: Number of neurons in the layer
        :param activation: Activation function
        :param model: Model to add layer
        '''
        if self.l2_reg is not None:
            return layer(n_neurons, activation=activation, kernel_regularizer=self.l2_reg)
        else:
            return layer(n_neurons, activation=activation)

    def add_fully_connected(self, model):
        '''
        Adds fully connected layers to the model (up to softmax layer)

        :param model: model to add layers
        '''
        fc = keras.layers.Conv3D(filters=512, kernel_size=1)(model)
        fc = keras.layers.MaxPooling3D(pool_size=2)(fc)
        fc = keras.layers.Flatten()(fc)

        # Adds FC layers excel softmax
        for i in range(0, self.fc_layers - 1):
            # Adds regularization if specified
            fc = self.__layer_regularized(keras.layers.Dense, self.fc_neurons, 'relu')(fc)

            # Adds dropout if specified
            if self.dropout > 0:
                fc = keras.layers.Dropout(self.dropout)(fc)

        # Adds softmax layers
        fc = self.__layer_regularized(keras.layers.Dense, self.n_classes, 'softmax')(fc)

        return fc

    def get_model(self):

        # Loads model based on stream type
        if self.stream_type == 's':

            imagenet = self.spatial_weights == 'imagenet'
            inputs, outputs = self.__get_cnn('s', imagenet)

            if os.path.isfile(self.spatial_weights):
                mf.load_weights(outputs, self.spatial_weights)

        elif self.stream_type == 't':
            inputs, outputs = self.__get_cnn('t', False)

            if os.path.isfile(self.temporal_weights):
                mf.load_weights(outputs, self.temporal_weights)

        else:

            # Checks if both streams weighs were provided
            #if self.spatial_weights is None or not os.path.isfile(self.spatial_weights) or self.temporal_weights is None or not os.path.isfile(self.temporal_weights):
            #    raise Exception('both spatial and temporal weights files must be provided to train this model!')

            s_inputs, s_outputs = self.__get_cnn('s', False)
            # mf.load_weights(spatial, self.spatial_weights)
            # for layer in spatial.layers:
            #    layer.trainable = False

            t_inputs, t_outputs = self.__get_cnn('t', False)
            #mf.load_weights(temporal, self.temporal_weights)
            # for layer in temporal.layers:
            #    layer.trainable = False

            concat = keras.layers.concatenate([s_outputs, t_outputs])
            fc = self.add_fully_connected(concat)

            return keras.Model(inputs=[s_inputs, t_inputs], outputs=fc)

        fc = self.add_fully_connected(outputs)
        return keras.Model(inputs=inputs, outputs=fc)


if __name__ == '__main__':

    model = TSCNN(
        cnn_model=CNNType.VGG16,
        stream_type='st',
        n_classes=101,
        fc_layers=3,
        fc_neurons=4096,
        chunks=5,
        nb_frames=10,
        dropout=.85,
        l2=1e-5,
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
