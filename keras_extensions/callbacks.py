import keras
import math
import keras_extensions.model_funcs as mf


class CustomCheckpointCallback(keras.callbacks.Callback):

    def __init__(self, filename, save_best_only):
        super().__init__()
        self.filename = filename
        self.save_best_only = save_best_only
        self.best_loss = math.inf

    def on_epoch_end(self, epoch, logs=None):
        if self.save_best_only:
            if logs['val_loss'] < self.best_loss:
                mf.save_model(self.model, self.filename)
                print('Validation loss improved from %f to %f and was saved in: %s' % (self.best_loss, logs['val_loss'], self.filename))
                self.best_loss = logs['val_loss']
        else:
            mf.save_model(self.model, self.filename)
            print('Model with val_loss %f saved in: %s' % (logs['val_loss'], self.filename))
