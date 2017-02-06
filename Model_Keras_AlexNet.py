
# coding: utf-8

# In[2]:

# AlexNet copied from:
# https://github.com/dandxy89/ImageModels/blob/master/AlexNet.py

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils.visualize_util import plot
from KerasLayers.Custom_layers import LRN2D

NB_CLASS = 1000         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'tf'

def build_model(weights_path=None): 
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 224, 224)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (224, 224, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Convolution Net Layer 1
    model = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    model = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(model)
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    model = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    model = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(model)
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(model)

    # Channel 1 - Convolution Net Layer 3
    model = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    model = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(model)
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(model)

    # Channel 1 - Convolution Net Layer 4
    model = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    model Channel 1 - Convolution Net Layer 5
    model = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(model)

    # Channel 1 - Cov Net Layer 6
    model = conv2D_lrn2d(model, 128, 27, 27, subsample=(1, 1), border_mode='same')
    model = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    model = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 7
    model = Flatten()(model)
    model = Dense(2048, activation='relu')(model)
    model = Dropout(DROPOUT)(model)

    # Channel 1 - Cov Net Layer 8
    model = Dense(2048, activation='relu')(model)
    model = Dropout(DROPOUT)(model)

    # Final Channel - Cov Net 9
    model = Dense(output_dim=NB_CLASS,
              activation='softmax')(model)
    
    if weights_path:
        model.load_weights(weights_path)
    return model, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


# In[ ]:



