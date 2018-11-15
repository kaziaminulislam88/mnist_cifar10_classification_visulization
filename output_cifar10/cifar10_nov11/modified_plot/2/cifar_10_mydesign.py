# Simple CNN model for CIFAR-10
import scipy.io as sio

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import numpy as np

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inpus from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
X_validation=X_test[0:100,:,:,:]
X_test1=X_test[100:,:,:,:]

y_validation=y_test[0:100]
y_test1=y_test[100:]
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test1)
num_classes = y_test1.shape[1]
y_validation = np_utils.to_categorical(y_validation)
# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=( 32, 32,3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test1, y_test1, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('model_cifar10_mydesign_new_way_nov12.h5')
model.save_weights('model_weights_cifar10_mydesign_new_way_nov12.h5')

y_test_arg=np.argmax(y_test1,axis=1);
sio.savemat('y_test_cifar10_new_way_nov12.mat', {'y_test1': y_test1})
sio.savemat('y_test_arg_cifar10_new_way_nov12.mat', {'y_test_arg': y_test_arg})
y_pred=model.predict(X_test1,verbose=0)

sio.savemat('y_pred_cifar10_new_way_nov12.mat', {'y_pred': y_pred})
y_pred_arg=np.argmax(y_pred,axis=1);
sio.savemat('y_pred_arg_cifar10_new_way_nov12.mat', {'y_pred_arg': y_pred_arg})

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
import numpy as np

from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_3')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])


for tv_weight in [1e-3, 1e-2, 1e-1, 1, 10]:
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.),
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])



for output_idx in np.arange(10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx,tv_weight=10, filter_indices=output_idx, input_range=(0., 1.))
    plt.figure()
    plt.title('Networks perception of {}'.format(output_idx))
    plt.imshow(img[..., 0])


fig = plt.figure()
for output_idx in np.arange(10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number=441+output_idx
    img = visualize_activation(model, layer_idx,tv_weight=100, filter_indices=output_idx, input_range=(0., 1.))

    plt.title('Networks perception of {}'.format(output_idx))
    plt.subplot(number)

    plt.imshow(img[..., 0])
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
for output_idx in np.arange(0,10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number=431+output_idx
    img = visualize_activation(model, layer_idx,tv_weight=100, filter_indices=output_idx, input_range=(0., 1.))

    plt.title('Networks perception of {}'.format(output_idx))
    plt.subplot(number)

    plt.imshow(img)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=1)
for output_idx in np.arange(0, 10):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number = 431 + output_idx
    plt.subplot(number)
    img = visualize_activation(model, layer_idx, tv_weight=100, filter_indices=output_idx, input_range=(0., 1.))
    plt.title('Networks perception of {}'.format(output_idx))

    plt.imshow(img)
plt.show()



# for layer 0
#preprocessing conv2d_5


layer_idx = utils.find_layer_idx(model, 'dense_3')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

# This is the output node we want to maximize.
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
plt.imshow(img[..., 0])


img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.))
plt.imshow(img[..., 0])

img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.), verbose=True)
plt.imshow(img[..., 0])

img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.),
                           tv_weight=0., lp_norm_weight=0., verbose=True)
plt.imshow(img[..., 0])

for tv_weight in [1e-3, 1e-2, 1e-1, 1, 10]:
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(0., 1.),
                               tv_weight=tv_weight, lp_norm_weight=0.)
    plt.figure()
    plt.imshow(img[..., 0])

#findal output
fig = plt.figure()
fig.subplots_adjust(hspace=1)
for output_idx in np.arange(0, 32):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number = 661 + output_idx
    plt.subplot(number)
    img = visualize_activation(model, layer_idx, filter_indices=output_idx, input_range=(0., 1.))

    # plt.title('Networks perception of {}'.format(output_idx+1))


    plt.imshow(img)
plt.show()



fig = plt.figure()
#fig.subplots_adjust(hspace=1)
for output_idx in np.arange(0, 32):
    # Lets turn off verbose output this time to avoid clutter and just see the output.
    number = 660 + output_idx
    plt.subplot(number)
    img = visualize_activation(model, layer_idx=3, filter_indices=output_idx, input_range=(0., 1.),tv_weight=1e-2, lp_norm_weight=0.)
    # plt.title('Networks perception of {}'.format(output_idx+1))
    plt.imshow(img)
plt.show()


for layer_idx1 in range(0,19):
    fig = plt.figure()
    #fig.subplots_adjust(hspace=1)
    for output_idx in np.arange(0, 32):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        number = 430 + output_idx
        ax = fig.add_subplot(6, 6, output_idx + 1)
        img = visualize_activation(model, layer_idx=layer_idx1, filter_indices=output_idx, input_range=(0., 1.),tv_weight=1e-2, lp_norm_weight=0.)
        # plt.title('Networks perception of {}'.format(output_idx+1))
        ax.imshow(img[..., 0])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("cifar_10_layer_number_"+str(layer_idx) + '.jpg')

os.chdir('/home/russel/PycharmProjects/')
import os
os.chdir('/home/russel/PycharmProjects/machine learning2/')
from keras.models import load_model
model=load_model('model_cifar10_mydesign_new_way_nov12.h5')

for layer_idx1 in range(18, 19):
    fig = plt.figure()
    # fig.subplots_adjust(hspace=1)
    for output_idx in np.arange(0, 10):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        # number = 430 + output_idx
        ax = fig.add_subplot(4, 3, output_idx + 1)
        img = visualize_activation(model, layer_idx=layer_idx1, filter_indices=output_idx, input_range=(0., 1.),
                                   tv_weight=100, lp_norm_weight=0.)
        plt.title('Networks perception of {}'.format(output_idx))
        ax.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("cifar10_layer_number_" + str(layer_idx1) + '_new_design_nov12.jpg')

for layer_idx1 in range(0,19):
    fig = plt.figure()
    #fig.subplots_adjust(hspace=1)
    for output_idx in np.arange(0, 32):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        number = 430 + output_idx
        ax = fig.add_subplot(6, 6, output_idx + 1)
        img = visualize_activation(model, layer_idx=layer_idx1, filter_indices=output_idx, input_range=(0., 1.),tv_weight=1e-1, lp_norm_weight=0.)
        # plt.title('Networks perception of {}'.format(output_idx+1))
        ax.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("cifar_10_layer_number_"+str(layer_idx) + '.jpg')

for layer_idx1 in range(0, 18):
    fig = plt.figure()
    # fig.subplots_adjust(hspace=1)
    for output_idx in np.arange(0, 32):
        # Lets turn off verbose output this time to avoid clutter and just see the output.
        number = 430 + output_idx
        ax = fig.add_subplot(6, 6, output_idx + 1)
        tv_weight1 = 1e-2
        img = visualize_activation(model, layer_idx=layer_idx1, filter_indices=output_idx, input_range=(0., 1.),
                                   tv_weight=tv_weight1, lp_norm_weight=0.)
        # plt.title('Networks perception of {}'.format(output_idx+1))
        ax.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()
    fig.savefig("cifar10_layer_number_" + str(layer_idx1) + '_tv_weights_' + str(tv_weight1) + '_new_design_nov11.jpg')

