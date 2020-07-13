import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as L
from classification_models.keras import Classifiers
ResNet18, preprocess_input = Classifiers.get('resnet18')
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import load_model, Model
import efficientnet.tfkeras as efn


def decoder_100x100(code_size):
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(3 * 3 * 64, activation='elu'))
    decoder.add(L.Reshape([3, 3, 64]))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='elu', padding='valid'))
    decoder.add(L.Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    return decoder

def decoder_112x112(code_size):
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(3 * 3 * 64, activation='elu'))
    decoder.add(L.Reshape([3, 3, 64]))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='valid'))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation='sigmoid', padding='same'))
    return decoder


def build_deep_autoencoder(conv_base, code_size):
    x = Flatten(name='flat')(conv_base.output)
    x = Dropout(0.5, name='drop1')(x)
    x = Dense(code_size, activation='elu', name='dense1')(x)
    encoder = Model(conv_base.input, x)

    decoder = decoder_100x100(code_size)
    return encoder, decoder

class CellOutput(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """
    def __init__(self):
        super(CellOutput, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        clear_output()


