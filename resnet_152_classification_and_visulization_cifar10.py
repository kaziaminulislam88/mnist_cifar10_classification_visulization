import os
os.chdir('D:/machine learning2/cnn_finetune-master/')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
import os.path
import h5py
import time
#from keras.utils import plot_model
import scipy as scipy
#!/usr/bin/env python
"""Trains a ResNet on the CIFAR10 dataset.
Greater than 91% test accuracy (0.52 val_loss) after 50 epochs
48sec per epoch on GTX 1080Ti
Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""
#link: https://github.com/fchollet/keras/blob/master/examples/cifar10_resnet.py
#from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import os
import cv2
# Load the mnist data.


import sys
sys.path.insert(0,'D:/machine learning2/cnn_finetune-master/')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
import os.path
import h5py
import time
# -*- coding: utf-8 -*-
import h5py
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import scipy.io as sio
from sklearn.metrics import log_loss
import os
from custom_layers.scale_layer import Scale
from resnet_152 import resnet152_model
from load_cifar10 import load_cifar10_data
from load_cifar10_testing import load_cifar10_data_testing

import resnet_152
img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 10
batch_size = 8
nb_epoch = 10



epochs = 10
batchSize = 1024
#trainingScenario = "balanced_20100319_1030010004B92000_056030364010"
trainingScenario='result'
numOfClasses = 10

# Load Cifar10 data. Please implement your own load_data() module for your own dataset
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

# Load our model
model = resnet152_model(img_rows, img_cols, channel, num_classes)
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
#np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
# Start Fine-tuning
hist=model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )

train_loss=hist.history['loss']
#val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
#val_acc=hist.history['val_acc']
xc=range(nb_epoch)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

img_rows=224
img_cols=224
num_channel=3
num_epoch=epochs
num_classes = 10

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])



model.save('resnet152_cifar10_modified_0ct16.h5');
# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
sio.savemat('predictions_valid_resnet152_cifar_oct16.mat', {'predictions_valid': predictions_valid})
# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)
sio.savemat('score_resnet152_cifar.mat', {'score': score})
model_predictions_training = model.predict(X_train)
sio.savemat('model_predictions_training_resnet152_cifar_oct16.mat', {'model_predictions_training': model_predictions_training})

#load testing data
X_train, Y_train, X_test, Y_test = load_cifar10_data_testing(img_rows, img_cols)
X_test2=np.rollaxis(X_test, 1,4);
X_test=X_test2;
predictions_testing = model.predict(X_test, batch_size=batch_size, verbose=1)
sio.savemat('predictions_testing_resnet152_cifar_oct16.mat', {'predictions_testing': predictions_testing})
sio.savemat('X_test_resnet152_cifar_oct16.mat', {'X_test': X_test})
sio.savemat('Y_test_resnet152_cifar_oct16.mat', {'Y_test': Y_test})
model.save_weights('resnet152_cifar10_modified_0ct16_weights.h5')

#test_image2=np.rollaxis(test_image, 2);
#test_image3=np.rollaxis(test_image2, 2);

test_image=X_test[0,:,:,:]

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
else:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)


print((model.predict(test_image)))

#print(model.predict_classes(test_image))

def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=0
filter_num=0
activations = get_featuremaps(model, int(layer_num),test_image)
print (np.shape(activations))
feature_maps = activations[0][0]
print (np.shape(feature_maps))

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')


num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

sio.savemat('Y_test_resnet152_cifar_oct17.mat', {'Y_test': Y_test})


Y_pred = model.predict(X_test)
sio.savemat('y_pred_resnet152_cifar_oct17.mat', {'Y_pred': Y_pred})
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
# y_pred = model.predict_classes(X_test)
# print(y_pred)
sio.savemat('y_pred_arg_resnet152_cifar_oct17.mat', {'y_pred': y_pred})
Y_test_arg=np.argmax(Y_test);
sio.savemat('Y_test_arg_resnet152_cifar_oct17.mat', {'Y_test_arg': Y_test_arg})



'''

for image_numb in range(10,20):
    test_image=X_test[image_numb,:,:,:]
    
    if num_channel == 1:
        if K.image_dim_ordering() == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
        else:
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
    else:
        if K.image_dim_ordering() == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
        else:
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
    
    
    print((model.predict(test_image)))
    
    #print(model.predict_classes(test_image))
    
    def get_featuremaps(model, layer_idx, X_batch):
        get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
        activations = get_activations([X_batch,0])
        return activations
    
    layer_num=0
    filter_num=0
    activations = get_featuremaps(model, int(layer_num),test_image)
    print (np.shape(activations))
    feature_maps = activations[0][0]
    print (np.shape(feature_maps))
    
    fig=plt.figure(figsize=(16,16))
    plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
    plt.savefig('image_'+str(image_numb)+'_original_class_'+str(np.argmax(Y_test[image_numb]))+"featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')
    
    
    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(16,16))
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
        ax = fig.add_subplot(subplot_num, subplot_num, i+1)
        #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(feature_maps[:,:,i],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig('image_'+str(image_numb)+'_original_class_'+str(np.argmax(Y_test[image_numb]))+"featuremaps-layer-{}".format(layer_num) + '.jpg')    
    
'''

''''

for layernum in range(0,10):
    image_numb=9
    test_image=X_test[image_numb,:,:,:]

    if num_channel == 1:
        if K.image_dim_ordering() == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
        else:
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
    else:
        if K.image_dim_ordering() == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)
        else:
            test_image = np.expand_dims(test_image, axis=0)
            print(test_image.shape)


    print((model.predict(test_image)))

    #print(model.predict_classes(test_image))

    def get_featuremaps(model, layer_idx, X_batch):
        get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
        activations = get_activations([X_batch,0])
        return activations

    layer_num=layernum
    filter_num=0
    activations = get_featuremaps(model, int(layer_num),test_image)
    print (np.shape(activations))
    feature_maps = activations[0][0]
    print (np.shape(feature_maps))

    fig=plt.figure(figsize=(16,16))
    plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
    plt.savefig('image_'+str(image_numb)+'_original_class_'+str(np.argmax(Y_test[image_numb]))+"featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')


    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(16,16))
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
        ax = fig.add_subplot(subplot_num, subplot_num, i+1)
        #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
        ax.imshow(feature_maps[:,:,i],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig('image_'+str(image_numb)+'_original_'+str(np.argmax(Y_test[image_numb]))+'_predicted_'+str(np.argmax( model.predict(test_image), axis=1))+"featuremaps-layer-{}".format(layer_num) + '.jpg')    
'''''