
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
#sys.path.insert(0,'D:/machine learning2/cnn_finetune-master/')
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



epochs = 100
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

# Start Fine-tuning


model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          )

# importing required libraries
from keras.models import Sequential
from scipy.misc import imread
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
from keras.datasets import mnist
#from cifar 10
#from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from scipy.misc import toimage, imresize
import numpy as np
#import resnet
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint
import scipy.io as sio
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
I_R=224;
X_train_original_1=x_train[0:3000,:,:,:];   #x_train.shape(50000, 32, 32, 3)
X_train_original_2=x_train[3000:6000,:,:,:];
X_train = np.zeros((X_train_original_1.shape[0], I_R, I_R, 3))
for i in range(X_train_original_1.shape[0]):
    X_train[i] = imresize(X_train_original_1[i], (I_R, I_R, 3), interp='bilinear', mode=None)
X_validation= np.zeros((X_train_original_2.shape[0], I_R, I_R, 3))
for i in range(X_train_original_2.shape[0]):
    X_validation[i] = imresize(X_train_original_2[i], (I_R, I_R, 3), interp='bilinear', mode=None)
Y_train = np_utils.to_categorical(y_train[0:3000])
Y_validation = np_utils.to_categorical(y_train[3000:6000])
x_test1=x_test[0:3000,:,:,:];  # x_test.shape(10000, 32, 32, 3)
X_test = np.zeros((x_test1.shape[0], I_R, I_R, 3))
for i in range(x_test1.shape[0]):
    X_test[i] = imresize(x_test1[i], (I_R, I_R, 3), interp='bilinear', mode=None)
Y_test1=y_test[0:3000];
Y_test = np_utils.to_categorical(Y_test1)

y_validation1=y_train[3000:6000]
sio.savemat('cifar10_resnet152_X_validation_Nov4.mat', {'X_validation': X_validation})
sio.savemat('cifar10_resnet152_X_test_Nov4.mat', {'X_test': X_test})

sio.savemat('cifar10_resnet152_y_validation1_Nov4.mat', {'y_validation1': y_validation1})
sio.savemat('cifar10_resnet152_Y_test1_Nov4.mat', {'Y_test1': Y_test1})

model_resnet152=model;
layer_name = 'flatten_2'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

intermediate_predict_validation = intermediate_layer_model.predict(X_validation)
intermediate_predict_test = intermediate_layer_model.predict(X_test)



sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_validation_Nov4.mat', {'intermediate_predict_validation': intermediate_predict_validation})
sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_test_Nov4.mat', {'intermediate_predict_test': intermediate_predict_test})

#bn5c_branch2a

layer_name = 'bn5c_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

intermediate_predict_validation = intermediate_layer_model.predict(X_validation)
intermediate_predict_test = intermediate_layer_model.predict(X_test)



sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_validation_bn5c_branch2a_Nov4.mat', {'intermediate_predict_validation': intermediate_predict_validation})
sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_test_bn5c_branch2a_Nov4.mat', {'intermediate_predict_test': intermediate_predict_test})


#res4b4_branch2a


layer_name = 'res4b4_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

intermediate_predict_validation = intermediate_layer_model.predict(X_validation)
intermediate_predict_test = intermediate_layer_model.predict(X_test)

sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_validation_res4b4_branch2a_Nov4.mat', {'intermediate_predict_validation': intermediate_predict_validation})
sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_test_res4b4_branch2a_Nov4.mat', {'intermediate_predict_test': intermediate_predict_test})

#res3a_branch2a

layer_name = 'res3a_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

intermediate_predict_validation = intermediate_layer_model.predict(X_validation)
intermediate_predict_test = intermediate_layer_model.predict(X_test)

sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_validation_res3a_branch2a_Nov4.mat', {'intermediate_predict_validation': intermediate_predict_validation})
sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_test_res3a_branch2a_Nov4.mat', {'intermediate_predict_test': intermediate_predict_test})

#res2a_branch2a

layer_name = 'res2a_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

intermediate_predict_validation = intermediate_layer_model.predict(X_validation)
intermediate_predict_test = intermediate_layer_model.predict(X_test)

sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_validation_res2a_branch2a_Nov4.mat', {'intermediate_predict_validation': intermediate_predict_validation})
sio.savemat('cifar10_resnet152_finetuned_intermediate_predict_test_res2a_branch2a_Nov4.mat', {'intermediate_predict_test': intermediate_predict_test})


#prediction_last_layer
predict_validation = model.predict(X_validation)
predict_test = model.predict(X_test)


sio.savemat('cifar10_resnet152_finetuned_predict_validation_Nov4.mat', {'predict_validation': predict_validation})
sio.savemat('cifar10_resnet152_finetuned_predict_test_Nov4.mat', {'predict_test': predict_test})

predict_test_arg=np.argmax(predict_test,axis=1)
predict_validation_arg=np.argmax(predict_validation,axis=1)

sio.savemat('cifar10_resnet152_finetuned_predict_test_arg_Nov4.mat', {'predict_test_arg': predict_test_arg})
sio.savemat('cifar10_resnet152_finetuned_predict_validation_arg_Nov4.mat', {'predict_validation_arg': predict_validation_arg})