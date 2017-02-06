
# coding: utf-8

# In[4]:

# Comma.ai model
def build_model(weights_path=None): 
    
    rows=14 
    cols=64
    ch=3
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 0.5,
            input_shape=(rows, cols, ch),
            output_shape=(rows, cols, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


# In[ ]:



