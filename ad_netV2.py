
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Lambda,MaxPooling2D,Dropout,Conv2DTranspose,concatenate,Input,Conv2D
from tensorflow.keras import Model
print(tf.__version__)


# In[2]:


def inblock(inp):

    print(inp)
    #inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    shortcut = Conv2D(32,(1,1),activation='elu', padding='same')(inp)
    
    conv3x3 = Conv2D(6, (3, 3),activation='elu', padding='same')(inp)# 32filters of size 3x3
    
    conv3x3 = tf.keras.layers.BatchNormalization(axis=3)(conv3x3)
    conv5x5 = Conv2D(12,(3, 3),activation='elu', padding='same')(conv3x3)# 32filters of size 3x3

    conv5x5 = tf.keras.layers.BatchNormalization(axis=3)(conv5x5)
    conv7x7 = Conv2D(14, (3, 3),activation='elu', padding='same')(conv5x5)#32filters of size 3x3

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    
    

    out = tf.keras.layers.add([shortcut, out])
    #out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("inblock",out.shape)
    return out


# In[3]:


def resblock_A(inp):
  
        inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
        shortcut = Conv2D(64,(1,1),activation='elu', padding='same')(inp)
    
        B1 = Conv2D(32, (1, 1),activation='elu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = Conv2D(32, (1, 1),activation='elu', padding='same')(inp)# 32filters of size 1x1
    
        B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
        B2 = Conv2D(32, (3, 3),activation='elu', padding='same')(B2)# 32filters of size 1x1
    
        B3 = Conv2D(32, (1, 1),activation='elu', padding='same')(inp)# 32filters of size 1x1
        
        B3 = tf.keras.layers.BatchNormalization(axis=3)(B3)
        B3 = Conv2D(48, (3, 3),activation='elu', padding='same')(B3)# 32filters of size 1x1
    
        B3 = tf.keras.layers.BatchNormalization(axis=3)(B3)
        B3 = Conv2D(64, (3, 3),activation='elu', padding='same')(B3)# 32filters of size 1x1
    
        out = tf.keras.layers.concatenate([B1, B2, B3], axis=3)
        out = Conv2D(64, (1, 1),activation='elu', padding='same')(out)
    
        out = tf.keras.layers.add([shortcut, out]) 
        out = tf.keras.layers.BatchNormalization(axis=3)(out)
        out = tf.keras.layers.Activation('elu')(out)
        #out = tf.keras.layers.BatchNormalization(axis=3)(out)
        print("resblock_a",out.shape)
        return out


# In[4]:


def reduction_A(inp):
    
    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    pooling = tf.keras.layers.MaxPooling2D((3,3),2)(inp)
    
    B1 = Conv2D(64, (3, 3),strides=(2,2),activation='elu', padding='valid')(inp)
    
    B2 = Conv2D(64, (1, 1),activation='elu', padding='same')(inp)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(96, (3, 3),activation='elu', padding='same')(B2)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(96, (3, 3),strides=(2,2),activation='elu', padding='valid')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    out = Conv2D(128, (1, 1),activation='elu', padding='same')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[5]:


def resblock_B(inp):

    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    shortcut = Conv2D(256,(1,1),activation='elu', padding='same')(inp)
    
    B1 = Conv2D(192, (1, 1), activation='elu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(128, (1, 1), activation='elu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(160, (1 , 7), activation='elu', padding='same')(B2)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(192, (7, 1), activation='elu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = Conv2D(256, (1, 1),activation='elu', padding='same')(out)
    
    out = tf.keras.layers.add([shortcut, out])  
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_B",out.shape)
    return out
    


# In[6]:


def reduction_B(inp):

    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    pooling = tf.keras.layers.MaxPooling2D((3,3),(2,2))(inp)
    
    B1 = Conv2D(64, (1, 1),activation='elu', padding='same')(inp)
    
    B1 = Conv2D(64, (3, 3),strides=(2,2),activation=None, padding='valid')(B1)
    
    B3 = Conv2D(64, (1, 1),activation='elu', padding='same')(inp)
    B3 = tf.keras.layers.BatchNormalization(axis=3)(B3)
    B3 = Conv2D(64, (3, 3),strides=(2,2),activation=None, padding='valid')(B3)
    
    B2 = Conv2D(64, (1, 1),activation='elu', padding='same')(inp)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(96, (3, 3),activation='elu', padding='valid')(B2)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(96, (3, 3),strides=(2,2),activation='relu', padding='valid')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2, pooling], axis=3)
    out = Conv2D(128, (1, 1),activation='elu', padding='same')(out)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    #print(out.shape)
    return out


# In[7]:


def resblock_C(inp):

    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    shortcut = Conv2D(1024,(1,1),activation='elu', padding='same')(inp)
    B1 = Conv2D(192, (1, 1), activation='elu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = Conv2D(192, (1, 1), activation='elu', padding='same')(inp)# 32filters of size 1x1
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(224, (1, 3), activation='elu', padding='same')(B2)
    
    B2 = tf.keras.layers.BatchNormalization(axis=3)(B2)
    B2 = Conv2D(256, (3, 1), activation='elu', padding='same')(B2)
    
    out = tf.keras.layers.concatenate([B1, B2], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = Conv2D(1024, (1, 1),activation='elu', padding='same')(out)
    
    out = tf.keras.layers.add([shortcut, out]) 
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("resblock_c",out.shape)
    return out


# In[8]:


def Path_3(inp):

    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    shortcut = Conv2D(32, (1, 1), activation='elu', padding='same')(inp)

    out = Conv2D(32, (3, 3), activation='elu', padding='same')(inp)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = Conv2D(32, (3, 3), activation='elu', padding='same')(out)

    out = tf.keras.layers.add([shortcut, out])
    #out = tf.keras.layers.concatenate([shortcut, out], axis=3)
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    out = tf.keras.layers.Activation('elu')(out)
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)

    print("path3",out.shape)

    return out


# In[9]:


def Path_4(inp):
  
    #out = conv_2d(inp, 32, 3, 3, activation='relu', padding='valid')
    inp = tf.keras.layers.BatchNormalization(axis=3)(inp)
    out = Conv2D(32,3,activation='elu',padding='same',strides=(1,1))(inp)
   
    out = tf.keras.layers.BatchNormalization(axis=3)(out)
    
    out = tf.keras.layers.Activation('elu')(out)
    
    #out = tf.keras.layers.BatchNormalization(axis=3)(out)
    print("path4",out.shape)

    

    return out


# In[10]:


def AD_net_V2(input_dimension=(256,256,3)):
 
    inputs = Input(input_dimension)
    inblock_inp= inblock(inputs)
    resblock_a=resblock_A(inblock_inp)
    path3=Path_3(resblock_a)
    
    #reduction_a=reduction_A(resblock_a)
    reduction_a=tf.keras.layers.MaxPooling2D((2,2))(resblock_a)
    resblock_b=resblock_B(reduction_a)
    path4=Path_4(resblock_b)
    
    #reduction_b=reduction_A(resblock_b)
    reduction_b=tf.keras.layers.MaxPooling2D((2,2))(resblock_b)
    resblock_c=resblock_C(reduction_b)
    
    
    Up1=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(resblock_c),path4],axis=3)
    resblock_b1=resblock_B(Up1)
   
    Up2=tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(resblock_b),path3],axis=3)
    resblock_a1=resblock_A(Up2)
    outblock=inblock(resblock_a1)
    
    conv_final=Conv2D(1,(1,1),activation='sigmoid',padding='same')(outblock)
    
    print("conv final",conv_final.shape) 
    
    model = tf.keras.Model(inputs=[inputs], outputs=[conv_final])
    model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    return model
    


# In[11]:


#@tf.function
def main():
    model=AD_net_V2()
    print(model.summary())


# In[ ]:


if __name__ == '__main__':
    main()

