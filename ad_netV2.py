
# coding: utf-8

# In[16]:


import tensorflow as tf
from tensorflow.keras.layers import Lambda,MaxPooling2D,Dropout,Conv2DTranspose,concatenate,Input,Conv2D
from tensorflow.keras import Model
print(tf.__version__)


# In[2]:


def inblock(inp):

    print(inp)
    shortcut = Conv2D(32,(1,1),activation='relu', padding='same')(inp)

    conv3x3 = Conv2D(6, (3, 3),activation='relu', padding='same')(inp)# 32filters of size 3x3

    conv5x5 = Conv2D(12,(3, 3),activation='relu', padding='same')(conv3x3)# 32filters of size 3x3

    conv7x7 = Conv2D(14, (3, 3),activation='relu', padding='same')(conv5x5)#32filters of size 3x3

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    
    

    out = tf.keras.layers.add([shortcut, out])
    #out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    
    out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("inblock",out.shape)
    return out


# In[18]:


def resblock_A(inp):
  
        shortcut = Conv2D(64,(1,1),activation='relu', padding='same')(inp)
    
        B1 = Conv2D(32, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = Conv2D(32, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = Conv2D(32, (3, 3),activation='relu', padding='same')(B2)# 32filters of size 1x1
    
        B3 = Conv2D(32, (1, 1),activation='relu', padding='same')(inp)# 32filters of size 1x1
    
        B3 = Conv2D(48, (3, 3),activation='relu', padding='same')(B3)# 32filters of size 1x1
    
        B3 = Conv2D(64, (3, 3),activation='relu', padding='same')(B3)# 32filters of size 1x1
    
        out = tf.keras.layers.concatenate([B1, B2, B3], axis=3)
        out = Conv2D(64, (1, 1),activation='relu', padding='same')(out)
    
        out = tf.keras.layers.add([shortcut, out])              
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        print("resblock_a",out.shape)
        return out


# In[34]:


def reduction_A(inp):

    pooling = tf.keras.layers.MaxPooling2D((3,3),2)(inp)
    
    B1 = Conv2D(64, (3, 3),strides=(2,2),activation='relu', padding='valid')(inp)
    
    B2 = Conv2D(64, (1, 1),activation='relu', padding='same')(inp)
    
    B2 = Conv2D(96, (3, 3),activation='relu', padding='valid')(B2)
    
    B2 = Conv2D(96, (3, 3),strides=(2,2),activation='relu', padding='valid')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    out = Conv2D(128, (1, 1),activation='relu', padding='same')(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[20]:


def resblock_B(inp):

    shortcut = Conv2D(256,(1,1),activation='relu', padding='same')(inp)
    
    B1 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(128, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(160, (1 , 7), activation='relu', padding='same')(B2)
    
    B2 = Conv2D(192, (7, 1), activation='relu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = Conv2D(256, (1, 1),activation='relu', padding='same')(out)
    
    out = tf.keras.layers.add([shortcut, out])  
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_B",out.shape)
    return out
    


# In[31]:


def reduction_B(inp):

    pooling = tf.keras.layers.MaxPooling2D((3,3),(2,2))(inp)
    
    B1 = Conv2D(64, (1, 1),activation='relu', padding='same')(inp)
    B1 = Conv2D(64, (3, 3),strides=(2,2),activation=None, padding='valid')(B1)
    
    B3 = Conv2D(64, (1, 1),activation='relu', padding='same')(inp)
    B3 = Conv2D(64, (3, 3),strides=(2,2),activation=None, padding='valid')(B3)
    
    B2 = Conv2D(64, (1, 1),activation='relu', padding='same')(inp)
    
    B2 = Conv2D(96, (3, 3),activation='relu', padding='valid')(B2)
    
    B2 = Conv2D(96, (3, 3),strides=(2,2),activation='relu', padding='valid')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    out = Conv2D(128, (1, 1),activation='relu', padding='same')(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[22]:


def resblock_C(inp):

    shortcut = Conv2D(1024,(1,1),activation='relu', padding='same')(inp)
    B1 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(192, (1, 1), activation='relu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(224, (1, 3), activation='relu', padding='same')(B2)
    
    B2 = Conv2D(256, (3, 1), activation='relu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = Conv2D(1024, (1, 1),activation='relu', padding='same')(out)
    
    out = tf.keras.layers.add([shortcut, out]) 
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_c",out.shape)
    return out


# In[42]:


def Path_3(inp):


    shortcut = Conv2D(32, (1, 1), activation='relu', padding='same')(inp)

    out = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    out = Conv2D(32, (3, 3), activation='relu', padding='same')(out)

    out = tf.keras.layers.add([shortcut, out])
    #out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)

    print("path3",out.shape)

    return out


# In[43]:


def Path_4(inp):
  
    #out = conv_2d(inp, 32, 3, 3, activation='relu', padding='valid')
    out = Conv2D(32,3,activation='relu',padding='same',strides=(1,1))(inp)
   

    #out = tf.keras.layers.Activation('relu')(out)
    
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("path4",out.shape)

    

    return out



