
from __future__ import print_function

import numpy as np
import keras

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(Dropout(0.1))
print(model.output)
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(Dropout(0.1))
print(model.output)
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(Dropout(0.1))
print(model.output)
print(model.output)

model.add(Flatten())
model.add(Dense(100, activation='relu'))
print(model.output)
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
print(model.output)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist_my_design_keras_vis_nov12.h5')
#visulization

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_2')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])


fig = plt.figure()
fig.subplots_adjust(hspace=1)
for output_idx in np.arange(0, 10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number = 431 + output_idx
    plt.subplot(number)
    img = visualize_activation(model, layer_idx, tv_weight=10, filter_indices=output_idx, input_range=(0., 1.))
    plt.title('Networks perception of {}'.format(output_idx))
    plt.imshow(img[..., 0])
plt.show()

for layer_idx1 in range(0, 13):
    fig = plt.figure()
    # fig.subplots_adjust(hspace=1)
    for output_idx in np.arange(0, 32):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        number = 430 + output_idx
        ax = fig.add_subplot(6, 6, output_idx + 1)
        img = visualize_activation(model, layer_idx=layer_idx1, filter_indices=output_idx, input_range=(0., 1.),
                                   tv_weight=1e-1, lp_norm_weight=0.)
        # plt.title('Networks perception of {}'.format(output_idx+1))
        ax.imshow(img[..., 0])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("mnist_layer_number_" + str(layer_idx1) + '_new_design.jpg')

