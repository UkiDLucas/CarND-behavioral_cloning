
# coding: utf-8

# In[1]:

"""
Wrapper methods around "import csv" that
are very specific to p3.
Created by Uki D. Lucas on Feb. 4, 2017
"""


# In[2]:

# snapping actual values to given labels

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# TEST
#assert (find_nearest(steering_labels, -0.951) == -1.00),"method find_nearest() has problem"


# In[3]:


import numpy as np

# TODO implement batch_from, batch_to - I did not need it for 8037 rows
# TODO implement has_header_row
def read_csv(file_path):
    """
    Usage:
    headers, data = read_csv(file_path)
    Parameter: 
    - file_path: can be relative path "../../../DATA/stuff.csv"
    Returns:
    - headers: array of strings e.g. ['steering', 'throttle', 'brake', 'speed']
    - data: array of strings, you have to convert values to int, float yourself
   test_read_csv()
    """
    import csv
    # Opening spreadsheet to read in TEXT mode: 'rt'
    with open(file_path, 'rt') as csvfile:
        # Most common format of CSV, TODO improve
        payload = csv.reader(csvfile, delimiter=',', quotechar='"') 
        row_counter = 0
        headers = []
        data = []
        for row in payload:
            row_counter = row_counter + 1
            
            if row_counter == 1:
                headers = row
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
            elif 1 < row_counter < 3:
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
                data.append(row)
            else:
                data.append(row)
        print("imported rows", row_counter)
        # I am returning data as numpy array instead of list
        # because it is handier to use it.
        return headers, np.array(data)
    
    
    
def test_read_csv():
    """
    This test is specific to Uki's enviroment.
    """
    data_dir = "../../../DATA/behavioral_cloning_data/"
    headers, data = read_csv(file_path = data_dir + 'driving_log.csv')
    print("headers \n",headers)
    print("3rd row of data \n",data[2:3])
# test_read_csv()


# In[4]:

import numpy as np
import math
    
def split_random(matrix, percent_train=70, percent_test=15):
    """
    Splits matrix data into randomly ordered sets 
    grouped by provided percentages.
    
    Usage:
    rows = 100
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = \
    split_random(matrix, percent_train=80, percent_test=10)
    
    percent_validation 10
    training (80, 2)
    testing (10, 2)
    validation (10, 2)
    
    Returns:
    - training_data: percentage_train e.g. 70%
    - testing_data: percent_test e.g. 15%
    - validation_data: reminder from 100% e.g. 15%
    Created by Uki D. Lucas on Feb. 4, 2017
    """
    #print(matrix)  
    row_count = matrix.shape[0]
    np.random.shuffle(matrix)
    
    end_training = int(math.ceil(row_count*percent_train/100))    
    end_testing = end_training + int(math.ceil((row_count * percent_test/100)))
    
    percent_validation = 100 - percent_train - percent_test
    
    training = matrix[:end_training]
    testing = []
    validation = []
    
    if percent_validation < 0:
        print("Make sure that the provided sum of " +         "training and testing percentages is equal, " +         "or less than 100%.")
        
        testing = matrix[end_training:]
    else:
        print("percent_validation", percent_validation)
        
        testing = matrix[end_training:end_testing]
        validation = matrix[end_testing:]
    
    return training, testing, validation

# TEST:

def test_split_random():
    rows = 8037
    columns = 2
    matrix = np.random.rand(rows, columns)
    training, testing, validation = split_random(matrix, percent_train=80, percent_test=20) 

    print("training",training.shape)
    print("testing",testing.shape)
    print("validation",validation.shape)
    
    print("sum",training.shape[0] + testing.shape[0])
    
    #print(split_random.__doc__)
# test_split_random()


# In[5]:

def get_image_center_values(matrix):
    data = [row[0] for row in matrix]
    return np.array(data)

def get_image_left_values(matrix):
    data = [row[1] for row in matrix]
    return np.array(data)

def get_image_right_values(matrix):
    data = [row[2] for row in matrix]
    return np.array(data)

def get_steering_values(matrix):
    data = [float(row[3]) for row in matrix]
    return np.array(data)

def get_throttle_values(matrix):
    data = [float(row[4]) for row in matrix]
    return np.array(data)

def get_brake_values(matrix):
    data = [float(row[5]) for row in matrix]
    return np.array(data)

def get_speed_values(matrix):
    data = [float(row[6]) for row in matrix]
    return np.array(data)


# In[6]:

def read_image(image_path):
    import cv2
    # cv2.IMREAD_COLOR 
    # cv2.COLOR_BGR2GRAY 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #print("image shape", image.shape)
    #plt.imshow(image, cmap='gray')
    #plt.show()
    return np.array(image)


# In[7]:

# for custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def false_rates(y_true, y_pred):
    false_neg = ...
    false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }


# In[8]:

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


# In[ ]:



