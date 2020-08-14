import keras


def reload_model(model):
    '''
    Saves the model into a Json and reload its structure
    '''
    model_json = model.to_json()
    return keras.models.model_from_json(model_json)


def save_model(model, path):
    '''
    Saves the model by freezing its trainable layers before. It is used to fix a bug on keras that the weights will
     not reload properly if saved while trainable is True
    '''

    # Registers which layer is trainable
    trainable = [layer.trainable for layer in model.layers]

    # Freeze layers to avoid bug when loading model back
    for layer in model.layers:
        layer.trainable = False

    # Save weights and reload model
    model.save_weights(path)

    # Reloads training status for each reloaded layer
    for i in range(0, len(model.layers)):
        model.layers[i].trainable = trainable[i]


def load_weights(model, weights_file):
    # Registers which layer is trainable
    trainable = [layer.trainable for layer in model.layers]

    model.load_weights(weights_file)

    # Reloads training status for each reloaded layer
    for i in range(0, len(model.layers)):
        model.layers[i].trainable = trainable[i]
