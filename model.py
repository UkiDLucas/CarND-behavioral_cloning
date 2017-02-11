
# coding: utf-8

# # model
# 
# Please refer to README file for project overview.

# In[1]:

nb_epoch = 5


data_dir = "../../../DATA/behavioral_cloning_data/"
processed_images_dir = "processed_images_64/"
image_final_width = 64
model_dir = "../../../DATA/MODELS/"
model_name = "model_p3_keras_tf_mini_14x64x3_"
batch_size = 256
driving_data_csv = "driving_log_no_zeros.csv"


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
headers, data = read_csv(data_dir + driving_data_csv)


# # Split data into training, testing and validation sets

# In[4]:

from DataHelper import split_random

# keras actually does it's own split, so here I just reserve small validation set.
training, testing, validation = split_random(data, percent_train=85, percent_test=15) 

print("training",training.shape)
print("testing",testing.shape)
print("validation",validation.shape)


# # Create Labels: steering value classes
# 
# - Please review notebook "preprocessing", section: "Steering value distribution".
# - Training labels have values ranging from -1 to +1.
# - When you steer with **keyboard** the STEPS are rather corse, so I think I can get away with **discrete steering angles, i.e. classes**.
# - I will start training with 21 equally spread classes, if needed I will increase to 41.
# - I want to make sure that my classes include **0.0 (zero)** as it is most common value.

# In[5]:

from DataHelper import plot_histogram, get_steering_values, find_nearest

steering_angles = get_steering_values(training)

change_step=0.01 # test data changes
plot_histogram("steering values", steering_angles, change_step)


# # Round steering angles
# 
# - I might consider rounding the steering angles to lower amount of training
# - I assume the classification labels to be float values

# In[6]:

import numpy as np
from numpy import ndarray

# desired number of classes, could be 21, too.
from DataHelper import create_steering_classes
steering_classes = create_steering_classes(number_of_classes = 41)

print("steering_classes", steering_classes)
number_of_classes = steering_classes.shape[0]
print("Number of classes", number_of_classes)

import matplotlib.pyplot as plt
plt.plot(steering_classes, 'b.')
plt.margins(0.1)
plt.title("Distribution of steering value classes.")
plt.xlabel("class number")
plt.ylabel('steering value')
plt.show()


# In[7]:



training_labels = np.array([], dtype=np.float32)

for raw_label in steering_angles:
    rounded_value = float( find_nearest(steering_classes, raw_label) )
    training_labels = np.append(training_labels, [rounded_value])
        
print(training_labels[0:50])


# In[8]:

change_step=0.01 # test data changes
plot_histogram("steering values", training_labels, change_step)


# # Encoding Training Labels in one-hot notation

# In[9]:

from DataHelper import encode_one_hot, locate_one_hot_position

y_one_hot =  encode_one_hot(defined_classes=steering_classes, sample_labels=training_labels)

print("y_one_hot", y_one_hot.shape)


# ## One-hot print and verify

# In[10]:

for index in range(10):
    print( index, ") \t", training_labels[index], "\t", y_one_hot[index],  
          "@", locate_one_hot_position(steering_classes, training_labels[index] ) )


# # Extract training features (images)

# In[11]:

from DataHelper import get_image_center_values 
image_names = get_image_center_values(training)
print(image_names.shape)
print(image_names[1])


# ## Create a list of image paths pointing to 64px version

# In[12]:

image_paths = []
for image_name in image_names:
    image_paths.extend([data_dir + processed_images_dir + image_name])
print(image_paths[1]) 


# In[13]:

from DataHelper import read_image
training_images = np.array([ read_image(path) for path in image_paths] )

print ("training_images matrix shape", training_images.shape)

plt.imshow(training_images[2], cmap='gray')
plt.show()


# In[14]:

from DataHelper import normalize_grayscale
training_features = normalize_grayscale(training_images) # you end up with -0.48823529 instead of -0.5
sample_image = training_features[2] #X_train[2]

plt.imshow(sample_image, cmap='gray')
plt.show()

show_rows = 1 # of 64
show_cols = 14
show_channels = 1 # of 3

print("sample_image \n", sample_image.shape,"\n", sample_image[:show_rows,:show_cols,:show_channels]) #  

def extract_image_single_channel(image):
    show_rows = image.shape[0]
    show_cols = image.shape[1]
    show_channels = 1 # red
    return np.array(  image[:show_rows, :show_cols, 0] )

# TEST
image_single_channel = extract_image_single_channel(sample_image)
print("image_single_channel: \n", image_single_channel[0][:18])
print("type and shape: \n", type(image_single_channel),image_single_channel.shape)print("training_images", training_images.shape)  

training_features = extract_image_single_channel(training_images)

print("single channel shape \n", training_features.shape)  
print("single channel column \n", training_features[0:4][0]) 
print("single channel pixel \n", training_features[0][0]) 
print("single channel red value \n", training_features[0][0][0])  
print("single channel red value \n", training_features[0][0][0])  

print("training_features", training_features.shape) 
# # Keras (with TensorFlow)
# 
# https://keras.io/layers/convolutional/

# In[15]:

import keras.backend as K
from keras.models import Sequential
from keras.layers import ELU
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda

from keras.activations import relu, softmax
from keras.optimizers import SGD
import cv2, numpy as np
from DataHelper import mean_pred, false_rates

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution1D


# ### See Model_Keras_VGG_16.py
# 
# This file (in the same directory) contains MODEL definiteion for VGG.16.
#from Model_Keras_VGG_16 import build_model # model = build_model('vgg16_weights.h5')
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16from keras.models import Model
from DataHelper import show_layers

model_VGG16 = VGG16(weights=None, include_top=True)

model_VGG16.summary()
number_of_layers = show_layers(model_VGG16)
# # Adjust VGG.16 model architecture to match my needs
print("number_of_layers", number_of_layers)

# create last layer with 21 classes
#x = Dense(21, activation='softmax', name='predictions')(model_VGG16.layers[-2].output)

# Convert to REGRESSION
last_layer = model_VGG16.layers[number_of_layers-1]
print("last_layer name",last_layer.name)

# One (1) output class makes this a (linear) regression.
x22 = Dense(1, activation='linear', name='regression')(last_layer.output)
model = Model(input=model_VGG16.input, output=x22)

model.summary()
from DataHelper import show_layers
show_layers(model)
# # Build custom sized model

# In[16]:

from keras.layers import InputLayer, Input

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu',
                        input_shape=(14, 64 ,3), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())


#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))


model.add(Dense(41, activation='linear')) # default: linear | softmax | relu | sigmoid
model.summary()


# # Compile model (configure learning process)

# In[17]:

# Before training a model, you need to configure the learning process, which is done via the compile method.
# 
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

optimizer='sgd' # | 'rmsprop'
loss_function="mse" # | 'binary_crossentropy' | 'mse' | mean_squared_error | sparse_categorical_crossentropy
metrics_array=['accuracy'] # , mean_pred, false_rates

model.compile(optimizer, loss_function, metrics_array)


# # Replace model with one stored on disk
# 
# - If you replace the model, the INPUT dimetions have to be the same as these trained
# - Name your models well

# In[18]:

from keras.models import load_model
model_path = model_dir + 'model_p3_keras_tf_mini_14x64x1a__epoch_35_val_loss_0.0200312916701.h5'
model = load_model(model_path) 
last_model_epochs = 35
model.summary()


# # Train (fit) the model agaist given labels

# In[19]:

history = model.fit(training_features, y_one_hot, nb_epoch=nb_epoch, 
                    batch_size=batch_size, verbose=1, validation_split=0.2)

Epoch 4/4
5464/5464 [==============================] - 60s - loss: 0.0307 - acc: 0.6067 - val_loss: 0.0288 - val_acc: 0.6130
# In[20]:

# list all data in history
print(history.history.keys())

training_accuracy = str( history.history['acc'][nb_epoch-1])
print("training_accuracy", training_accuracy)

training_error = str( history.history['loss'][nb_epoch-1])
print("training_error", training_error)

validation_accuracy = str( history.history['val_acc'][nb_epoch-1])
print("validation_accuracy", validation_accuracy)

validation_error = str( history.history['val_loss'][nb_epoch-1])
print("validation_error", validation_error)


# # Save the model

# In[21]:

# creates a HDF5 file '___.h5'
model.save(model_dir + model_name + "_epoch_" + str(nb_epoch + last_model_epochs) 
           + "_val_loss_" + str(validation_error) + ".h5") 
#del model  # deletes the existing model
#model = load_model('my_model.h5')


# In[22]:

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy (bigger better)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'testing accuracy'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Validation error (smaller better)')
plt.ylabel('error')
plt.xlabel('epochs run')
plt.legend(['training error(loss)', 'validation error (loss)'], loc='upper right')
plt.show()


# # Prediction set

# In[23]:

image_name = "IMG/center_2016_12_01_13_32_43_659.jpg" # stering 0.05219137
original_steering_angle = 0.05219137

#image_name = "IMG/center_2016_12_01_13_33_10_579.jpg" # 0.1287396

#image_name = "IMG/center_2016_12_01_13_39_28_024.jpg" # -0.9426954
#original_steering_angle = -0.9426954

image_path =  data_dir + processed_images_dir + image_name
print(image_path)
image = read_image(image_path)
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()


# In[24]:

#expected convolution2d_input_1 to have 4 dimensions, but got array with shape (14, 64, 3)
image = image[None, :, :]

import keras
#from keras.np_utils import probas_to_classes

predictions = model.predict( image, batch_size=1, verbose=1)
print(type(predictions), predictions.shape, predictions)
prediction = float(predictions[0][0])
#print("prediction \n", prediction)
#print("np.argmax(prediction) ",np.argmax(prediction))
most_likely = np.argmax(predictions)
print("np.argmax(predictions) ", most_likely )


# In[25]:

# summarize history for loss
plt.plot(predictions[0])
plt.title('predictions')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['predictions'], loc='upper right')
plt.show()


# In[26]:

print("original steering angle", original_steering_angle)
print("predicted steering angle", steering_classes[most_likely])


# In[ ]:




# In[ ]:



