
# coding: utf-8

# # Simplest Keras (TensorFlow) model
# 
# To get comfortable with Keras (using TensorFlow), I had to create the simples model possible.
# In this experiment I simply predict if the number is SMALL (from 0 to 0.5) or BIG (from 0.5 to 1).
# Since there can be an **infinite number of real values between 0 to 1, it is actually a GOOD CHALLENGE for the machine learning WITHOUT telling computer the rules I have used to classify my numbers**.
# 
# - I auto-generate a lot of random numbers between 0 and 1.
# - I label them SMALL or BIG (not telling computer how or why).
# - I let computer look at my numbers and their classification,
# - then, I create new numbers that computer did not see before
# - The computer tries to make a prediction.
# 
# ### ***The ability to make predicion based on data without knowing underlaying rules is the real power on machine learning***

# ### Features
# 
# - If you imagine a spreadsheet, the **row is a single data sample**.
# - Each **data sample (row) has features that describe it**.
# - You can think about samples as e.g. **personnal data**,
# - the **features would be weight, height, age, smoking** or not.
# - In machine learning it is common to use data set with hundreds, or thousands features, much more than a human brain can comprehend.

# In[1]:

FEATURES_PER_SAMPLE = 4 # The provided data has to have constant amount of feature columns 


# ### Import necessary Python libraries

# In[2]:

import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation

from keras.activations import relu, softmax
from keras.optimizers import SGD


# ### A little function to make numbers more human friendly

# In[3]:

def convert_to_human(label):
    if label < 0.5:
        return "small"
    else:
        return "big"


# ### Function to generate random data

# In[4]:

def generate_random_features(number_of_samples, features_per_sample): 
    import numpy as np
    features_matrix = np.random.rand(number_of_samples, features_per_sample)
    return np.array(features_matrix)

# test function
features = generate_random_features(number_of_samples = 100000, features_per_sample = FEATURES_PER_SAMPLE)
print(features[:5]) # show first few ..


# ### Function that takes generated data and assign labels (small|big) to each sample

# In[5]:

def generate_labels_for_features(features):
    """
    To make things simpler, 
    this function takes on FIRST piece of data "sample[0]" for considertion.
    """
    labels_list = []
    for sample in features:
        value = sample[0]
        if value < 0.5:
            labels_list.append([0.0]) # small numbers
        else:
            labels_list.append([1.0]) # big numbers
    return np.array(labels_list)

# test function
labels = generate_labels_for_features(features)
print(labels[:5])  # show first few ..


# # Machine Learning model (very simple)

# In[6]:

model = Sequential()
output_dim = 1
model.add(Dense(output_dim, input_shape=(FEATURES_PER_SAMPLE,), activation='sigmoid', name="001_Dense"))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ## Training the model agaist the generated data

# In[7]:

model.fit(features, labels, nb_epoch=5, batch_size=100)


# ## Generate testing data that computer did not see during the training

# In[8]:

test_features = generate_random_features(number_of_samples = 100, features_per_sample = FEATURES_PER_SAMPLE)
test_labels = generate_labels_for_features(test_features)


# ## Running predicitons

# In[9]:

predictions = model.predict_classes(test_features, batch_size=10, verbose=1)

print()
for index in range(len(test_features[:10])):
    tested_value = test_features[index,0]
    expected_label = test_labels[index]
    print("%.2f" % tested_value, convert_to_human(expected_label),           "\t, but predicted: ", convert_to_human(predictions[index]), predictions[index] ) 


# ## Evaluating the success

# In[10]:

scores = model.evaluate(test_features, test_labels, verbose=1)
print("scores: \n", scores)

achieved_test_accuracy = scores[1]*100
print("Achieved test accuracy (%s): %.1f%%" % (model.metrics_names[1], achieved_test_accuracy ))
#cvscores.append(scores[1] * 100)


# ### Summary: I was able to achieve up to **98% accuracy** which is very good for such as simple model.

# In[ ]:



