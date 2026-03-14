# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:59:14 2022

@author: HP
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #to disable GPU

from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers import BatchNormalization, Dropout, SpatialDropout2D
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

save_path='E:/database/UASpeech/audio_sorted/resnet6/'

pid=['F02','F03','F04','F05','M01','M04','M05','M07','M08','M09','M10','M11','M12','M14','M16']



def _conv(**conv_params):
    """Helper to build a conv block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    trainable=conv_params.setdefault("trainable", True)

    def f(input):
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      trainable=trainable,
                      kernel_regularizer=kernel_regularizer)(input)

    return f


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))
    trainable=conv_params.setdefault("trainable", True)

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      trainable=trainable,
                      kernel_regularizer=kernel_regularizer)(input)
        norm = BatchNormalization(trainable=trainable)(conv)              
        return Activation("relu")(norm)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    trainable=conv_params.setdefault("trainable", True)

    def f(input):
        norm = BatchNormalization(trainable=trainable)(input)
        activation = Activation("relu")(norm)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      trainable=trainable,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

unfreeze=False
input = Input(shape=(150,150,1))
conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2), trainable=unfreeze)(input)
pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
block = pool1

conv1 = _conv(filters=64, kernel_size=(3, 3),strides=(1,1), trainable=unfreeze)(block)
residual = _bn_relu_conv(filters=64, kernel_size=(3, 3), trainable=unfreeze)(conv1)
block=add([block, residual])

conv1 = _bn_relu_conv(filters=128, kernel_size=(3, 3), strides=(2,2), trainable=unfreeze)(block)
residual = _bn_relu_conv(filters=128, kernel_size=(3, 3), trainable=unfreeze)(conv1)
shortcut = _conv(filters=128, kernel_size=(1, 1), strides=(2,2), padding="valid", trainable=unfreeze)(block)
block= add([shortcut, residual])

conv1 = _bn_relu_conv(filters=128, kernel_size=(3, 3), strides=(1,1), trainable=unfreeze)(block)
residual = _bn_relu_conv(filters=128, kernel_size=(3, 3), trainable=unfreeze)(conv1)
block=add([block, residual])

conv1 = _bn_relu_conv(filters=256, kernel_size=(3, 3), strides=(2,2))(block)
conv1 = SpatialDropout2D(0.7)(conv1)
residual = _bn_relu_conv(filters=256, kernel_size=(3, 3))(conv1)
shortcut = _conv(filters=256, kernel_size=(1, 1), strides=(2,2), padding="valid", trainable=unfreeze)(block)
block= add([shortcut, residual])

norm = BatchNormalization()(block)
block = Activation("relu")(norm)
pool2 = AveragePooling2D(pool_size=(10,10), strides=(1, 1))(block)
flatten1 = Flatten()(pool2)
flatten1=Dropout(0.5)(flatten1)

dense = Dense(units=155, kernel_initializer="he_normal", activation="softmax")(flatten1)
       
model = Model(inputs=input, outputs=dense)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=['accuracy'])

pix=5                                                                                                                                                                                                                                                                                                                                                           

model.load_weights('E:/database/UASpeech/audio_sorted/resnet6/hc_train/resnet6_asr.h5')  #best model
    
checkpoint = ModelCheckpoint(save_path+ pid[pix]+'_resnet6_wt.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

# main_path='E:/database/UASpeech/audio_sorted/patient/'
# main_path='E:/database/UASpeech/audio_sorted/Preprocessed_data2/'
main_path='E:/database/UASpeech/audio_sorted/vad_data/'
X_train = np.load(main_path+ pid[pix]+'_train_input.npy')
y_train = np.load(main_path+ pid[pix]+'_train_output.npy')
X_test = np.load(main_path+ pid[pix]+'_test_input.npy')
y_test = np.load(main_path+ pid[pix]+'_test_output.npy')
Y_train = np_utils.to_categorical(y_train, 155)
Y_test = np_utils.to_categorical(y_test, 155)

history = model.fit(X_train, Y_train, epochs=1000, batch_size=16, shuffle=True,
                    callbacks=[checkpoint], verbose=1, validation_data=(X_test, Y_test))

#result log
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

np.savetxt(save_path+pid[pix]+'_accuracy.csv', accuracy)
np.savetxt(save_path+pid[pix]+'_val_accuracy.csv', val_accuracy)

# csa=model.evaluate(X_train, Y_train, batch_size=16)
# print('csa=',csa[1]*100)

# osa=model.evaluate(X_test, Y_test, batch_size=16)
# print('osa=',osa[1]*100)
