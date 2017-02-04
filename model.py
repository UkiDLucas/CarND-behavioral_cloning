
# coding: utf-8

# # Training UkiNet model
# 
# Please refer to README file for project overview.

# In[1]:

data_dir = "../../../DATA/behavioral_cloning_data/"


# In[12]:

import CSV_helper
print(CSV_helper.__doc__)
from CSV_helper import test_read_csv, read_csv
print(read_csv.__doc__)
#test_read_csv()
# fetch actual log of driving data
headers, data = read_csv(data_dir + "driving_log.csv")


# # Labels (steering value classes)
# 
# - Please review notebook "preprocessing", section: "Steering value distribution".
# - Training labels have values ranging from -1 to +1.
# - When you steer with **keyboard** the STEPS are rather corse, so I think I can get away with **discrete steering angles, i.e. classes**.
# - I will start training with 21 equally spread classes, if needed I will increase to 41.
# - I want to make sure that my classes include **0.0 (zero)** as it is most common value.

# In[73]:

import numpy as np
from numpy import ndarray
number_of_classes = 21
steering_labels = np.linspace(-1, 1, num=number_of_classes, endpoint=True) 
steering_labels = np.sort(steering_labels)
print("steering_labels", steering_labels)
print("Number of classes",steering_labels.shape[0])

import matplotlib.pyplot as plt
plt.plot(steering_labels, 'b.')
plt.margins(0.1)
plt.title("Distribution of steering value classes.")
plt.xlabel("class number")
plt.ylabel('steering value')
plt.show()


# In[79]:

# snapping actual values to given labels

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# TEST
assert (find_nearest(steering_labels, -0.951) == -1.00),"method find_nearest() has problem"


# In[ ]:

# TODO: Load the feature data to the variable X_train
X_train = data['features']
# TODO: Load the label data to the variable y_train
y_train = data['labels']


# # Keras (with TensorFlow)

# In[13]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


# In[14]:

def UkiNet(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

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
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# 

# 
