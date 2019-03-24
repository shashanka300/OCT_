
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
from tensorflow.keras.layers import Input,add,Conv2D,Lambda
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


# In[2]:


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


# In[3]:


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


# In[4]:


#@tf.function
def inblock(inp):
    '''
    Block 1 input
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    
    shortcut = Conv2D(51,(1,1),activation='relu', padding='same')(inp)

    conv3x3 = Conv2D(8, (3, 3),activation='relu', padding='same')(inp)# 32filters of size 3x3

    conv5x5 = Conv2D(17,(3, 3),activation='relu', padding='same')(conv3x3)# 32filters of size 3x3

    conv7x7 = Conv2D(26, (3, 3),activation='relu', padding='same')(conv5x5)#32filters of size 3x3

    out = tf.keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    
    out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print("inblock",out.shape)
    return out


# In[5]:


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
    
        B1 = Conv2D(filter_size, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = Conv2D(filter_size, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = Conv2D(filter_size, (3, 3),activation='relu', padding='same')(B2)# 32filters of size 1x1
    
        B3 = Conv2D(filter_size, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B3 = Conv2D(48, (3, 3),activation='relu', padding='same')(B3)# 32filters of size 1x1
    
        B3 = Conv2D(64, (3, 3),activation='relu', padding='same')(B3)# 32filters of size 1x1
    
        out = tf.keras.layers.concatenate([B1, B2, B3], axis=3)
    
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
                       
        out = Conv2D(384, (1, 1),activation='relu', padding='same')(out)# 32filters of size 1x1  
    
        #out = add([inp, out])
        out = tf.keras.layers.concatenate([inp, out], axis=3)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        #print("resblock_a",out.shape)
        return out


# In[6]:


#@tf.function
def Path_1(inp):
    '''
    path from first layer to final
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = Conv2D(32, (1, 1),activation='relu', padding='same')(inp)

    out = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    out = Conv2D( 32, (3, 3), activation='relu', padding='same')(out)

    out = add([shortcut, out])
    out = tf.keras.layers.Activation('relu')(out)
    branch = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = Conv2D( 32, (3, 3), activation='relu', padding='same')(branch)
    out = Conv2D( 32, (3, 3), activation='relu', padding='same')(out)

    out = add([branch, out])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    #print("path",out.shape)
    return out


# In[7]:


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
    
    B1 = Conv2D(inp,384, 3, 3, strides=(2,2))
    
    B2 = Conv2D(inp,192, 1, 1)# 64filters of size 1x1  
    
    B2 = Conv2D(B2,224, 3, 3)# 64filters of size 1x1 
    
    B2 = Conv2D(B2,256, 3, 3,strides=(2,2))# 64filters of size 1x1 
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[8]:


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
    
    B1 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(128, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(160, (1 , 7), activation='relu', padding='same')(B2)
    
    B2 = Conv2D(192, (7, 1), activation='relu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = Conv2D(1154, (1, 1), activation='relu', padding='same')(out)
    
    #out = add([inp, out])
    out = tf.keras.layers.concatenate([inp, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print("resblock_B",out.shape)
    return out
    
    
    
    


# In[9]:


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
    
    B1 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(224, (1, 3), activation='relu', padding='same')(B2)
    
    B2 = Conv2D(256, (3, 1), activation='relu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = Conv2D(2048, (1, 1), activation='relu', padding='same')(out)
    
    #out = add([inp, out])
    out = tf.keras.layers.concatenate([inp, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print("resblock_c",out.shape)
    return out


# In[10]:


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
    
    B1 = Conv2D(inp, 256, 1, 1)
    B1 = Conv2D(B1, 384, 3, 3, strides=(2,2))
    
    B2 = Conv2D(inp,256, 1, 1)# 64filters of size 1x1
    B2 = Conv2D(B2, 288, 1, 1, strides=(2,2))
    
    B3 = Conv2D(inp, 256, 1, 1)# 64filters of size 1x1 
    B3 = Conv2D(B3, 288, 1, 1)
    B3 = Conv2D(B3, 320, 1, 1, strides=(2,2))
    
    out = tf.keras.layers.concatenate([pooling, B1, B2, B3], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[17]:


#@tf.function
def Path_2(inp):
    '''
    path for second layer
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = Conv2D(32, (1, 1), activation='relu', padding='same')(inp)

    out = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    out = Conv2D(32, (3, 3), activation='relu', padding='same')(out)

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = Conv2D(32, (3, 3), activation='relu', padding='same')(out)

    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print("path2",out.shape)

    return out


# In[23]:


#@tf.function
def Path_3(inp):
    '''
    path for third layer
    
    Arguments:
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = Conv2D(32, (1, 1), activation='relu', padding='same')(inp)

    out = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    out = Conv2D(32, (3, 3), activation='relu', padding='same')(out)

    #out = tf.keras.layers.add([shortcut, out])
    out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    #print("path3",out.shape)

    return out


# In[24]:


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
    out = Conv2D(32,3,activation='relu',padding='same',strides=(1,1))(inp)
   

    #out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print("path4",out.shape)

    

    return out


# In[28]:


#@tf.function
def AD_net(input_dimension=(256,256,3)):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''
    inputs = Input(input_dimension)
    inblock_inp= inblock(inputs)
    resblock_a1=resblock_A(inblock_inp,32)
    path1=Path_1(resblock_a1)
    
    #reduction_a=reduction_A(resblock_a1)
    reduction_a=tf.keras.layers.MaxPooling2D((2,2))(resblock_a1)
    resblock_a2=resblock_A(reduction_a,64)
    path2=Path_2(resblock_a2)
    
    reduction_b=tf.keras.layers.MaxPooling2D((2,2))(resblock_a2
                                                         )
    resblock_b1=resblock_B(reduction_b)
    path3=Path_3(resblock_b1)
    
    Mpooling=tf.keras.layers.MaxPooling2D((2,2))(resblock_b1)
    resblock_b2=resblock_B(Mpooling)
    path4=Path_4(resblock_b2)
    
    Mpooling2=tf.keras.layers.MaxPooling2D((2,2))(resblock_b2)
    resblock_c=resblock_C(Mpooling2)
    
    Up1=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(resblock_c),path4],axis=3)
    resblock_b2=resblock_B(Up1)

    Up2=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(resblock_b2),path3],axis=3)
    resblock_b3=resblock_B(Up2)

    
    Up3=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(resblock_b3),path2],axis=3)
    resblock_a3=resblock_A(Up3,64)
    
    
    
    Up4=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(resblock_a3),path1],axis=3)
    resblock_a4=resblock_A(Up4,32)
    outblock=inblock(resblock_a4)
    
    conv_final=Conv2D(1,(1,1),activation='sigmoid',padding='same')(outblock)
    #print("conv final",conv_final.shape) 
    
    model = tf.keras.Model(inputs=[inputs], outputs=[conv_final])
    #modle.compile(tf.keras.optimizer.Adam(lr),loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    return model

    
    


# In[29]:


#@tf.function
def main():
    model=AD_net()
    print(model.summary())


# In[30]:


if __name__ == '__main__':
    main()

