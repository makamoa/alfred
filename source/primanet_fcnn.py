from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Add, Activation, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 8

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False, dim=1):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (dim, None, None)
    else:
        input_shape = (None, None, dim)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        # Block 0
    x = Conv2D(4, (5, 5), activation='relu', padding='valid', name='block0_conv1')(img_input)
    #x = BatchNormalization(name='BatchNormal_1')(x)
    x = Activation('relu', name='act')(x)

    # Block 1
    x = Conv2D(8, (7, 7), activation='relu', padding='valid', name='block1_conv1')(x)
    x = Conv2D(8, (7, 7), activation='relu', padding='valid', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(16, (7, 7), activation='relu', padding='valid', name='block2_conv1')(x)
    x = Conv2D(16, (7, 7), activation='relu', padding='valid', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(32, (7, 7), activation='relu', padding='valid', name='block3_conv1')(x)
    x = Conv2D(32, (7, 7), activation='relu', padding='valid', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool2')(x)

    # Block 4
    x = Conv2D(64, (7, 7), activation='relu', padding='valid', name='block4_conv1')(x)
    x = Conv2D(64, (7, 7), activation='relu', padding='valid', name='block4_conv2')(x)
    return x
