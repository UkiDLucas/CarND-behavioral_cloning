
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

print("build_model v7")

def build_model(weights_path=None, image_width=224, image_height=224, color_channels=3, number_of_classes=21):

    model = Sequential()
    # got array with shape (5626, 224, 224, 3)
    # got array with shape (402, 224, 224, 3)
    model.add(ZeroPadding2D((1,1),input_shape=(image_width, image_height, color_channels), dim_ordering="tf"))
    #model.add(ZeroPadding2D((1,1),input_shape=(color_channels, image_width, image_height), dim_ordering="tf"))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
  
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="tf"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
  
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    
    # expected dense_3 to have shape (None, 1000) but got array with shape (402, 1)
    # expected dense_3 to have shape (None, 21) but got array with shape (402, 1)
    #model.add(Dense(number_of_classes, activation='softmax'))
    model.add(Dense(1, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# In[ ]:



