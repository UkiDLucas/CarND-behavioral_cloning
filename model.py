
# coding: utf-8

# # Training UkiNet model
# 
# Please refer to README file for project overview.

# In[1]:

data_dir = "../../../DATA/behavioral_cloning_data/"


# In[2]:


import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import cv2


# In[3]:

import DataHelper
#print(DataHelper.__doc__)
from DataHelper import test_read_csv, read_csv
#print(read_csv.__doc__)
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

# In[4]:

import numpy as np
from numpy import ndarray
number_of_classes = 21
steering_classes = np.linspace(-1, 1, num=number_of_classes, endpoint=True) 
steering_classes = np.sort(steering_classes)
print("steering_classes", steering_classes)
print("Number of classes",steering_classes.shape[0])

import matplotlib.pyplot as plt
plt.plot(steering_classes, 'b.')
plt.margins(0.1)
plt.title("Distribution of steering value classes.")
plt.xlabel("class number")
plt.ylabel('steering value')
plt.show()


# # Split data into training, testing and validation sets

# In[5]:

from DataHelper import split_random, get_image_center_values
image_center_values = get_image_center_values(data)

training, testing, validation = split_random(data, percent_train=70, percent_test=15) 

print("training",training.shape)
print("testing",testing.shape)
print("validation",validation.shape)


# # Extract training features (images)

# In[6]:

from DataHelper import get_image_center_values 
image_names = get_image_center_values(training)
print(image_names.shape)
print(image_names[1])


# ## Create a list of image paths pointing to 64px version

# In[7]:

image_paths = []
for image_name in image_names:
    image_paths.extend([data_dir + "processed_images_64/" + image_name])
print(image_paths[1]) 


# In[8]:

from DataHelper import read_image
training_features = np.array([ read_image(path) for path in image_paths] )

print ("matrix shape", training_features.shape)
plt.imshow(training_features[2], cmap='gray')
plt.show()


# In[ ]:




# In[9]:

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

training_features_normalized = normalize_grayscale(training_features)


# In[ ]:




# # Extract training labels (steering value classes) 

# In[10]:

from DataHelper import get_steering_values, find_nearest
raw_labels = get_steering_values(training)
print(raw_labels.shape)
print(raw_labels[0:5])

training_labels = np.array([], dtype=np.float64)

for raw_label in raw_labels: # raw_labels[:5]
    #print("raw_label", raw_label)
    label = find_nearest(steering_classes, raw_label)
    #print("normalized label", label)
    
    # Classification labels should be strings:
    training_labels = np.append(training_labels, [str(label)])
        
print(training_labels)


# ## One hot

# In[11]:

import sklearn
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(training_labels)
print(y_one_hot)


# # Extract 

# # Keras (with TensorFlow)

# In[12]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation

from keras.activations import relu, softmax
from keras.optimizers import SGD
import cv2, numpy as np


# In[57]:

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D
    
def UkiNet(weights_path=None):
    """
    The cropped camera images are expected to be (3, 14, 64), not square
    e.g.: 
    - training_features_normalized (5626, 14, 64, 3)
    - y_one_hot (5626, 18)
    
    https://keras.io/layers/convolutional/
    """
    model = Sequential()

    # IN: (samples, rows, cols, channels)'tf'
    model.add(Convolution2D(96, 14, 64, dim_ordering='tf', input_shape=(14, 64, 3), activation='relu', name="01_Conv2D"))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(96, 14, 64, dim_ordering='tf', input_shape=(14, 64, 3), activation='relu', name="02_Conv2D"))
    
    #ValueError: Negative dimension size caused by subtracting 14 from 1 for 'Conv2D_7' (op: 'Conv2D') 
    #    with input shapes: [?,1,1,96], [14,64,96,96].

    
    
    
    #model.add(ZeroPadding2D((1,1),input_shape=(1,64,64)))
    #model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3,14,64),name="02_Conv_ReLU_")) #
    model.add(Flatten(input_shape=(14, 64, 3), name="01_flatten_14x64x3")) 
    
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Dense(128, name="02_Dense_128")) # Fully connected 
    model.add(Activation('relu', name="03_ReLU")) # ReLU activation
    model.add(Dense(43, name="04_Dense_43"))
    model.add(Activation('softmax', name="05_Activation_Softmax"))
    
    #Input 0 is incompatible with layer flatten_7: expected ndim >= 3, found ndim=2
    #model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    
    # I want to end up with 21 classes
    model.add(Dense(18, activation='softmax', name="06_Dense_18"))

    if weights_path:
        model.load_weights(weights_path)
    return model


print("training_features_normalized", training_features_normalized.shape)
print("y_one_hot", y_one_hot.shape)
model = UkiNet()
# Configures the learning process and metrics
model.compile('sgd', 'mean_squared_error', ['accuracy'])
history = model.fit(training_features_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)


# In[ ]:




# In[ ]:



