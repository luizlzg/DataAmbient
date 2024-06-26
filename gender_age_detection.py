import os
import numpy as np
import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
else:
	from tensorflow import keras
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

#---------------------------------------

def baseModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))

	return model

#url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'

def VGGModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'):

	model = baseModel()

	#-----------------------------------

	output = './weights/vgg_face_weights.h5'

	if os.path.isfile(output) != True:
		print("vgg_face_weights.h5 will be downloaded...")
		gdown.download(url, output, quiet=False)

	#-----------------------------------
	
	model.load_weights(output)

	#-----------------------------------

	#TO-DO: why?
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

	return vgg_face_descriptor


def Age_Model():

    agemodel = Sequential()
    agemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(160, 160, 3)))
    agemodel.add(MaxPooling2D((2,2)))
    agemodel.add(Conv2D(64, (3,3), activation='relu'))
    agemodel.add(MaxPooling2D((2,2)))
    agemodel.add(Conv2D(128, (3,3), activation='relu'))
    agemodel.add(MaxPooling2D((2,2)))
    agemodel.add(Flatten())
    agemodel.add(Dense(64, activation='relu'))
    agemodel.add(Dense(3, activation='softmax'))

    agemodel.load_weights('./weights/age_model_weights_mtcnn.h5')

    return agemodel

def Gender_Model(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5'):

	model = VGGModel()

	#--------------------------

	classes = 2
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	gender_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights


	if os.path.isfile('./weights/gender_model_weights.h5') != True:
		print("gender_model_weights.h5 will be downloaded...")

		output = './weights/gender_model_weights.h5'
		gdown.download(url, output, quiet=False)

	gender_model.load_weights('./weights/gender_model_weights.h5')

	return gender_model

	#--------------------------