# -*- coding: utf-8 -*-
import cv2
import numpy as np
import copy
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
import sys

sys.setrecursionlimit(3000)


class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)
        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma' % self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta' % self.name)
        self.trainable_weights = [self.gamma, self.beta]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)
    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)
    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def resnet152_model(weights_path=None):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    model = Model(img_input, x_fc)
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model


if __name__ == '__main__':

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'resnet152_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'resnet152_weights_tf.h5'
    # Insert a new dimension for the batch_size

    # Test pretrained model
    model = resnet152_model(weights_path)


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


model_resnet152=model;

output_predict_train = model.predict(X_train)
output_predict_validation = model.predict(X_validation)

output_predict_test = model.predict(X_test)

sio.savemat('cifar10_resnet152_output_predict_train_Nov1.mat', {'output_predict_train': output_predict_train})
sio.savemat('cifar10_resnet152_output_predict_validation_Nov1.mat', {'output_predict_validation': output_predict_validation})
sio.savemat('cifar10_resnet152_output_predict_test_Nov1.mat', {'output_predict_test': output_predict_test})






layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)


from keras.layers import Dense, Activation
intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name+'cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name+'cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test':intermediate_test})

layer_name = 'bn5c_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

from keras.layers import Dense, Activation

intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name + 'cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name + 'cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test': intermediate_test})

#res4b4_branch2a
layer_name = 'res4b4_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

from keras.layers import Dense, Activation

intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name + '_cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name + '_cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test': intermediate_test})


#res3a_branch2a
layer_name = 'res3a_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

from keras.layers import Dense, Activation

intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name + '_cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name + '_cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test': intermediate_test})

#res2a_branch2a

layer_name = 'res2a_branch2a'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

from keras.layers import Dense, Activation

intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name + '_cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name + '_cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test': intermediate_test})


#conv1

layer_name = 'conv1'
intermediate_layer_model = Model(inputs=model_resnet152.input, outputs=model_resnet152.get_layer(layer_name).output)

from keras.layers import Dense, Activation

intermediate_train = intermediate_layer_model.predict(X_train)
intermediate_test = intermediate_layer_model.predict(X_test)

sio.savemat(layer_name + '_cifar10_resnet152_output_intermediate_train_Nov1.mat', {'intermediate_train': intermediate_train})
sio.savemat(layer_name + '_cifar10_resnet152__intermediate_test_Nov1.mat', {'intermediate_test': intermediate_test})