
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
from tensorflow.keras.layers import Input,add
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


# In[2]:


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)


# In[34]:


#@tf.function
def conv_2d(x,filters, num_row, num_col, padding='same', strides=(1, 1), activation=None, name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
    
    
    Returns:
        [keras layer] -- [output layer]
    '''
  
    inp_C = tf.keras.layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, activation=activation, use_bias=False)(x)
    inp_C = tf.keras.layers.BatchNormalization(axis=3, scale=False)(x)

    return inp_C


# In[35]:


#@tf.function
def trans_conv_2d(x,filters, num_row, num_col, padding='valid', strides=(1, 1), name=None):
    '''
    #On deconvolution first UnPooling then RELU and then deconvolution is done.
    2D Transposed Convolutional layers
    
    Arguments:
         x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    inp_D = tf.keras.layers.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    inp_D = tf.keras.layers.BatchNormalization(axis=3, scale=False)(x)
    
    return inp_D


# In[36]:


#@tf.function
def inblock(inp):
    '''
    Block 1 input
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    print(inp)
    shortcut = conv_2d(inp,51, 1, 1)# 32filters of size 1x1

    conv3x3 = conv_2d(shortcut,8, 3, 3)# 32filters of size 3x3

    conv5x5 = conv_2d(conv3x3,17, 3, 3)# 32filters of size 3x3

    conv7x7 = conv_2d(conv5x5,26, 3, 3)#32filters of size 3x3

    out = tf.keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    
    out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("inblock",out.shape)
    return out


# In[37]:


#@tf.function
def resblock_A(inp,filter_size):
        '''
    resblock input
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
        

        '''
     
       # inp = tf.keras.layers.Activation('relu')(inp)    
    
        B1 = conv_2d(inp,filter_size, 1, 1)# 32filters of size 1x1
    
        B2 = conv_2d(inp,filter_size, 1, 1)# 32filters of size 1x1
    
        B2 = conv_2d(B2,filter_size, 3, 3)# 32filters of size 1x1
    
        B3 = conv_2d(inp,filter_size, 1, 1)# 32filters of size 1x1
    
        B3 = conv_2d(B3,48, 3, 3)# 32filters of size 1x1
    
        B3 = conv_2d(B3,64, 3, 3)# 32filters of size 1x1
    
        out = tf.keras.layers.concatenate([B1, B2, B3], axis=3)
    
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
                       
        out = conv_2d(out,384, 1, 1)# 32filters of size 1x1  
    
        #out = add([inp, out])
        out = tf.keras.layers.concatenate([inp, out], axis=3)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        print("resblock_a",out.shape)
        return out


# In[38]:


#@tf.function
def Path_1(inp):
    '''
    path from first layer to final
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = conv_2d(inp, 32, 1, 1,
                         activation='relu')

    out = conv_2d(inp, 32, 3, 3, activation='relu', padding='same')
    out = conv_2d(out, 32, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    branch = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = conv_2d(branch, 32, 3, 3, activation='relu', padding='same')
    out = conv_2d(out, 32, 3, 3, activation='relu', padding='same')

    out = add([branch, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    print("path",out.shape)
    return out


# In[39]:


#@tf.function
def reduction_A(inp):
    '''
    reduction block A
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    pooling = tf.keras.layers.MaxPooling2D((3,3),2)(inp)
    
    B1 = conv_2d(inp,384, 3, 3, strides=(2,2))
    
    B2 = conv_2d(inp,192, 1, 1)# 64filters of size 1x1  
    
    B2 = conv_2d(B2,224, 3, 3)# 64filters of size 1x1 
    
    B2 = conv_2d(B2,256, 3, 3,strides=(2,2))# 64filters of size 1x1 
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[40]:


#@tf.function
def resblock_B(inp):
    '''
    resblock B
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    #inp = tf.keras.layers.Activation('relu')(inp) 
    
    B1 = conv_2d(inp,192, 1, 1)# 32filters of size 1x1
    
    B2 = conv_2d(inp,128, 1, 1)# 32filters of size 1x1
    
    B2 = conv_2d(B2,160, 1 , 7)
    
    B2 = conv_2d(B2,192, 7, 1)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = conv_2d(out,1154, 1, 1)
    
    #out = add([inp, out])
    out = tf.keras.layers.concatenate([inp, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_B",out.shape)
    return out
    
    
    
    


# In[58]:


#@tf.function
def resblock_C(inp):
    '''
    resblock C
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    #inp = tf.keras.layers.Activation('relu')(inp) 
    
    B1 = conv_2d(inp,192, 1, 1)# 32filters of size 1x1
    
    B2 = conv_2d(inp,192, 1, 1)# 32filters of size 1x1
    
    B2 = conv_2d(B2,224, 1, 3)
    
    B2 = conv_2d(B2,256, 3, 1)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = conv_2d(out,2048, 1, 1)
    
    #out = add([inp, out])
    out = tf.keras.layers.concatenate([inp, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_c",out.shape)
    return out


# In[59]:


#@tf.function
def reduction_B(inp):
    '''
    reduction block A
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    pooling = tf.keras.layers.MaxPooling2D((3,3),(2,2))(inp)
    
    B1 = conv_2d(inp, 256, 1, 1)
    B1 = conv_2d(B1, 384, 3, 3, strides=(2,2))
    
    B2 = conv_2d(inp,256, 1, 1)# 64filters of size 1x1
    B2 = conv_2d(B2, 288, 1, 1, strides=(2,2))
    
    B3 = conv_2d(inp, 256, 1, 1)# 64filters of size 1x1 
    B3 = conv_2d(B3, 288, 1, 1)
    B3 = conv_2d(B3, 320, 1, 1, strides=(2,2))
    
    out = tf.keras.layers.concatenate([pooling, B1, B2, B3], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print(out.shape)
    return out


# In[60]:


#@tf.function
def Path_2(inp):
    '''
    path for second layer
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = conv_2d(inp, 32, 1, 1,
                         activation='relu')

    out = conv_2d(inp, 32, 3, 3, activation='relu', padding='same')
    out = conv_2d(out, 32, 3, 3, activation='relu', padding='same')

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = conv_2d(out, 32, 3, 3, activation='relu', padding='same')

    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("path2",out.shape)

    return out


# In[61]:


#@tf.function
def Path_3(inp):
    '''
    path for third layer
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = conv_2d(inp, 32, 1, 1,
                         activation='relu')

    out = conv_2d(inp, 32, 3, 3, activation='relu', padding='same')
    out = conv_2d(out, 32, 3, 3, activation='relu', padding='same')

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    print("path3",out.shape)

    return out


# In[66]:


#@tf.function
def Path_4(inp):
    '''
    path for third layer
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    #out = conv_2d(inp, 32, 3, 3, activation='relu', padding='valid')
    out = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same',strides=(1,1))(inp)
   

    #out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("path4",out.shape)

    

    return out


