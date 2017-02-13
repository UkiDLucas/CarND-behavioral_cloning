
# coding: utf-8

# # model
# 
# Please refer to README file for project overview.

# # Set execution parameters

# In[1]:

data_dir = "../_DATA/CarND_behavioral_cloning/r_001/"
image_final_width = 64
driving_data_csv = "driving_log_normalized.csv"
processed_images_dir = "processed_images/"

model_dir = "../_DATA/MODELS/"
model_name = "model_p3_14x64x3_"
batch_size = 256
nb_epoch = 10 
# 30 epochs = 55 minutes on MacBook Pro

# CONTINUE TRAINING ?
should_retrain_existing_model = True
model_to_continue_training = "model_p3_keras_tf_mini_14x64x3__epoch_30_val_acc_0.402555912543.h5"
previous_trained_epochs = 30


# # Python Imports

# In[2]:

import matplotlib.pyplot as plt

# import matplotlib.image as mpimg
# from scipy import misc
# import cv2

import DataHelper
#print(DataHelper.__doc__)


# # Fetch CSV driving data

# In[3]:

from  DataHelper import read_csv

print(data_dir + driving_data_csv)
headers, data = read_csv(data_dir + driving_data_csv)


# # Split Training, Testing and Validation sets
# 
# Keras actually does it's own training/testing split, so here I just reserve small validation set.

# In[4]:

from DataHelper import split_random

training, testing, validation = split_random(data, percent_train=85, percent_test=15) 

print("training", training.shape)
print("testing", testing.shape)
print("validation", validation.shape)


# # Fetch steering angles

# In[5]:

from DataHelper import plot_histogram, get_steering_values, find_nearest

steering_angles = get_steering_values(training)

change_step=0.01 # test data changes
plot_histogram("steering values", steering_angles, change_step)


# # Create discrete 41 steering classes
# 
# - I might consider rounding the steering angles to lower amount of training
# - I assume the classification labels to be float values

# In[6]:

import numpy as np
from numpy import ndarray
from DataHelper import create_steering_classes, plot_steering_values

steering_classes = create_steering_classes(number_of_classes = 41).astype(np.float32)

# ROUNDING is a hopeless effort
# steering_classes = np.round(steering_classes, 2)

print("steering_classes", steering_classes, type(steering_classes[0]))
number_of_classes = steering_classes.shape[0]
print("Number of created classes", number_of_classes)

plot_steering_values(values = steering_classes)


# ## Snap the ACTUAL steering angles to newly created classes

# In[7]:

training_labels = np.array([], dtype=np.float32)

for actual_steering_angle in steering_angles:
    rounded_value = find_nearest(steering_classes, actual_steering_angle).astype(np.float32)
    training_labels = np.append(training_labels, [rounded_value])
        
print(training_labels[0:50], type(training_labels[0]))


# In[8]:

plot_histogram("steering values", training_labels, change_step=0.01)


# ## Test conversion form actual to class label

# In[9]:

from random import randrange

sample_index = randrange(0,len(steering_angles))
print("actual:", steering_angles[sample_index], "class:",training_labels[sample_index])

sample_index = randrange(0,len(steering_angles))
print("actual:", steering_angles[sample_index], "class:",training_labels[sample_index])

sample_index = randrange(0,len(steering_angles))
print("actual:", steering_angles[sample_index], "class:",training_labels[sample_index])


# ## Encoding Training Labels in one-hot notation

# In[10]:

from DataHelper import encode_one_hot, locate_one_hot_position

y_one_hot =  encode_one_hot(defined_classes=steering_classes, sample_labels=training_labels)

print("y_one_hot", y_one_hot.shape)


# ### One-hot print and verify

# In[11]:

for index in range(5):
    print( "training label", training_labels[index], "is @", 
          locate_one_hot_position(steering_classes, training_labels[index] ), 
          "\n", y_one_hot[index] )


# # Extract training features (images)

# In[12]:

from DataHelper import get_image_center_values 

image_names = get_image_center_values(training)
print(image_names.shape)
print(image_names[1])


# ## Create a list of image paths pointing to 64px version

# In[13]:

image_paths = []
for image_name in image_names:
    image_paths.extend([data_dir + processed_images_dir + image_name])
print(image_paths[1]) 


# ## Read actual (preprocessed) images from the disk

# In[14]:

from DataHelper import read_image

training_features = np.array([ read_image(path) for path in image_paths] )

print ("training_features matrix shape", training_features.shape)

sample_image = training_features[2]
plt.imshow(sample_image) # cmap='gray' , cmap='rainbow'
plt.show()

print(sample_image[0][0:15])


# ## X Extract single channel (red)
from DataHelper import normalize_grayscale

show_rows = 1 # of 64
show_cols = 14
show_channels = 1 # of 3

print("sample_image \n", sample_image.shape,"\n", sample_image[:show_rows,:show_cols,:show_channels]) #  def extract_image_single_channel(image):
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


# ### X Import Model_Keras_VGG_16.py
# 
# This file (in the same directory) contains MODEL definiteion for VGG.16.
#from Model_Keras_VGG_16 import build_model # model = build_model('vgg16_weights.h5')
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.models import Model
from DataHelper import show_layers

model_VGG16 = VGG16(weights=None, include_top=True)

model_VGG16.summary()
number_of_layers = show_layers(model_VGG16)
# ### X Adjust VGG.16 model architecture to match my needs
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
# # Build my own custom model

# In[16]:

from keras.layers import InputLayer, Input

# activation = "relu" | "elu"

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', activation="relu" ,
                        input_shape=(14, 64 ,3), dim_ordering='tf', name="conv2d_1_64x3x3_relu"))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation="relu", name="conv2d_2_128x3x3_relu" ))
model.add(Convolution2D(256, 5, 5, border_mode='same', activation="relu", name="conv2d_3_256x5x5_relu" ))

model.add(Flatten())

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(256, activation="relu", name="dense_1_256_relu"))
model.add(Dropout(0.25, name="dropout_1_0.25"))
model.add(Dense(256, activation="relu", name="dense_2_256_relu" ))

# CLASSIFICATION
model.add(Dense(41, activation='linear' , name="dense_3_41_linear")) # default: linear | softmax | relu | sigmoid

# REGRESSION
#model.add(Dense(1, activation='linear'))

model.summary()


# # Compile model (configure learning process)

# In[ ]:

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
from keras.models import load_model

if should_retrain_existing_model:
    model_path = model_dir + model_to_continue_training
    model = load_model(model_path) 
    model.summary()
# # Train (fit) the model agaist given labels

# In[ ]:

# REGRESSION
# history = model.fit(training_features, training_labels, nb_epoch=nb_epoch, 
#                    batch_size=batch_size, verbose=1, validation_split=0.2)

# CLASSIFICATION
history = model.fit(training_features, y_one_hot, nb_epoch=nb_epoch, 
                    batch_size=batch_size, verbose=1, validation_split=0.2)

Train on 2501 samples, validate on 626 samples
Epoch 1/5
2501/2501 [==============================] - 109s - loss: 0.0207 - acc: 0.3135 - val_loss: 0.0199 - val_acc: 0.3642
Epoch 2/5
2501/2501 [==============================] - 106s - loss: 0.0207 - acc: 0.3083 - val_loss: 0.0199 - val_acc: 0.3594
Epoch 3/5
2501/2501 [==============================] - 106s - loss: 0.0207 - acc: 0.3071 - val_loss: 0.0199 - val_acc: 0.3594
Epoch 4/5
2501/2501 [==============================] - 105s - loss: 0.0207 - acc: 0.3067 - val_loss: 0.0199 - val_acc: 0.3546
Epoch 5/5
2501/2501 [==============================] - 104s - loss: 0.0206 - acc: 0.3203 - val_loss: 0.0199 - val_acc: 0.3562

__
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 14, 64, 64)    1792        convolution2d_input_2[0][0]      
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 64, 128)   73856       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 14, 64, 256)   295168      convolution2d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 229376)        0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           58720512    flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 256)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 256)           65792       dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 41)            10537       dense_2[0][0]                    
====================================================================================================
Total params: 59,167,657
Trainable params: 59,167,657
Non-trainable params: 0
_________________________
# In[ ]:

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

# In[ ]:

# creates a HDF5 file '___.h5'
model.save(model_dir + model_name + "_epoch_" + str(nb_epoch + previous_trained_epochs) 
           + "_val_acc_" + str(validation_accuracy) + ".h5") 
#del model  # deletes the existing model
#model = load_model('my_model.h5')


# In[ ]:

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


# # Prediction
from keras.models import load_model

model_path = model_dir + model_to_continue_training
print(model_path)

model = load_model(model_dir + model_to_continue_training) 
model.summary()
# In[ ]:

image_name = "IMG/center_2016_12_01_13_32_43_659.jpg" # stering 0.05219137
original_steering_angle = 0.05219137

image_name = "IMG/center_2016_12_01_13_33_10_579.jpg" # 0.1287396
original_steering_angle = 0.05219137

image_name = "IMG/center_2016_12_01_13_39_28_024.jpg" # -0.9426954
original_steering_angle = -0.9426954

image_path =  data_dir + processed_images_dir + image_name
print(image_path)
image = read_image(image_path)
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()


# ## Run model.predict(image)

# In[ ]:

predictions = model.predict( image[None, :, :], batch_size=1, verbose=1)


# ## Extract top prediction

# In[ ]:

from DataHelper import predict_class

predicted_class = predict_class(predictions, steering_classes)

print("original steering angle \n", original_steering_angle)
print("top_prediction \n", predicted_class )


# ## Plot predictions (peaks are top classes)

# In[ ]:

# summarize history for loss
plt.plot(predictions[0])
plt.title('predictions')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['predictions'], loc='upper right')
plt.show()


# In[ ]:



