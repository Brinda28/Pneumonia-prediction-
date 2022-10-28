#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *


# In[ ]:


base_dir = 'C:/Users/purib/Downloads/archive (1)/chest_xray'

print('Contents of base directory:')
print(os.listdir(base_dir))

print('\nContents of train directory:')
print(os.listdir(f'{base_dir}/train'))

print('\nContents of val directory:')
print(os.listdir(f'{base_dir}/val'))

print('\nContents of test directory:')
print(os.listdir(f'{base_dir}/test'))


# In[ ]:


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

#Directory with training NORMAL/PNEUMONIA pictures
train_NORMAL_dir = os.path.join(train_dir, 'NORMAL')
train_PNEUMONIA_dir = os.path.join(train_dir, 'PNEUMONIA')

#Directory with validation NORMAL/PNEUMONIA pictures
val_NORMAL_dir = os.path.join(val_dir, 'NORMAL')
val_PNEUMONIA_dir = os.path.join(val_dir, 'PNEUMONIA')

#Directory with test NORMAL/PNEUMONIA pictures
test_NORMAL_dir = os.path.join(test_dir, 'NORMAL')
test_PNEUMONIA_dir = os.path.join(test_dir, 'PNEUMONIA')


# In[30]:


train_NORMAL_fnames = os.listdir(train_NORMAL_dir)
train_PNEUMONIA_fnames = os.listdir(train_PNEUMONIA_dir)

print(train_NORMAL_fnames[:5])
print(train_PNEUMONIA_fnames[:5])


# In[31]:


nrows = 2
ncols = 4

pic_index = 0 

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*2)

pic_index+=4

next_NORMAL_pix = [os.path.join(train_NORMAL_dir, fname) 
                for fname in train_NORMAL_fnames[pic_index-4:pic_index]]

next_PNEUMONIA_pix = [os.path.join(train_PNEUMONIA_dir, fname) 
                for fname in train_PNEUMONIA_fnames[pic_index-4:pic_index]]

for i, img_path in enumerate(next_NORMAL_pix+next_PNEUMONIA_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img, cmap='gray')

plt.show();


# In[32]:


print('total training NORMAL images :', len(os.listdir(train_NORMAL_dir)))
print('total training PNEUMONIA images :', len(os.listdir(train_PNEUMONIA_dir)))

print('total validation NORMAL images :', len(os.listdir(val_NORMAL_dir)))
print('total validation PNEUMONIA images :', len(os.listdir(val_PNEUMONIA_dir)))

print('total test NORMAL images :', len(os.listdir(test_NORMAL_dir)))
print('total test PNEUMONIA images :', len(os.listdir(test_PNEUMONIA_dir)))


# In[33]:


ax = sns.barplot(x = ['NORMAL', 'PNEUMONIA'], y = [len(os.listdir(train_NORMAL_dir)), len(os.listdir(train_PNEUMONIA_dir))]
            , palette = 'Set1')
for p in ax.patches:
    ax.annotate(format(p.get_height()), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, -12), 
                   textcoords = 'offset points')
plt.xlabel('Target', size=14)
plt.ylabel('Count', size=14);


# In[34]:


train_datagen = ImageDataGenerator(rescale = 1.0/255., 
                                   featurewise_center=False,  # set input mean to 0 over the dataset
                                   samplewise_center=False,  # set each sample mean to 0
                                   featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                   samplewise_std_normalization=False,  # divide each input by its std
                                   zca_whitening=False,  # apply ZCA whitening
                                   rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
                                   zoom_range = 0.2, # Randomly zoom image 
                                   width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                   height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                   horizontal_flip = True,  # randomly flip images
                                   vertical_flip=False)
val_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen  = ImageDataGenerator(rescale = 1.0/255.)


#Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 128,
                                                    class_mode = 'binary',
                                                    target_size = (299, 299),
                                                    color_mode='grayscale')     

#Flow validation images in batches of 20 using val_datagen generator

val_generator =  val_datagen.flow_from_directory(val_dir,
                                                 batch_size = 8,
                                                 class_mode = 'binary',
                                                 target_size = (299, 299),
                                                 color_mode='grayscale')

#Flow test images in batches of 20 using test_datagen generator

test_generator =  test_datagen.flow_from_directory(test_dir,
                                                   batch_size = 64,
                                                   class_mode = 'binary',
                                                   target_size = (299, 299),
                                                   color_mode='grayscale')


# In[35]:


#Initialize the base model.
#Set the input shape and remove the dense layers.
pre_trained_model = InceptionV3(input_shape = (299, 299, 3), include_top = False)

#Freeze the weights of the layers.
for layer in pre_trained_model.layers:
  layer.trainable = False


# In[36]:


model = tf.keras.models.Model(inputs = pre_trained_model.inputs, outputs = pre_trained_model.get_layer('mixed7').output)

input_tensor = Input(shape=(299,299,1))
x = Conv2D(3,(3,3),padding='same')(input_tensor)
x = model(x)
x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation = 'sigmoid')(x)
model = tf.keras.models.Model(inputs = input_tensor, outputs = out)


# In[37]:


model.summary()


# In[38]:


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[39]:


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#Set the training parameters
model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', 
              metrics = ['accuracy', f1_m, precision_m, recall_m])


# In[40]:


history = model.fit(train_generator,
                    steps_per_epoch = 10,
                    epochs = 15,
                    validation_data = val_generator,
                    validation_steps = 2,
                    verbose = 1)


# In[41]:


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  (epochs, acc, 'bo', color = '#ff0066')
plt.plot  (epochs, val_acc, color = '#00ccff')
plt.title ('Training and validation accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  (epochs, loss, 'bo', color = '#ff0066')
plt.plot  (epochs, val_loss, color = '#00ccff')
plt.legend(['train', 'val'], loc='upper left')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title ('Training and validation loss');


# In[42]:


model.evaluate(test_generator)


# In[ ]:





# In[ ]:





# In[ ]:




