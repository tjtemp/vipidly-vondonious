from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.layers.core import Dropout
#from prep_dataset import load_train, load_test

from keras.models import Sequential


def vgg_16(img_rows, img_cols, img_channels):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block1_conv1', input_shape=(img_rows, img_cols, img_channels)))
	model.add(Convolution2D(64, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block1_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))


	model.add(Convolution2D(128, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block2_conv1'))
	model.add(Convolution2D(128, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block2_conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))


	model.add(Convolution2D(256, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block3_conv1'))
	model.add(Convolution2D(256, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block3_conv2'))
	model.add(Convolution2D(256, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block3_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))


	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block4_conv1'))
	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block4_conv2'))
	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block4_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))


	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block5_conv1'))
	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block5_conv2'))
	model.add(Convolution2D(512, 3, 3, activation='relu', init='he_normal',
		border_mode='same', name='block5_conv3'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))


	model.add(Flatten(name='flatten'))
	model.add(Dense(4096, activation='relu', name='fc1'))
	model.add(Dense(2048, activation='relu', name='fc2'))
	model.add(Dense(1024, activation='relu', name='fc3'))
	model.add(Dropout(0.25))
	model.add(Dense(512, activation='relu', name='fc4'))
	model.add(Dense(256, activation='relu', name='fc5'))
	model.add(Dense(128, activation='relu', name='fc6'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu', name='fc7'))
	model.add(Dense(32, activation='relu', name='fc8'))
	model.add(Dense(10, activation='softmax', name='predictions'))

	return model

# 18, 34, 50, 101, 152
from resnet import *
def resnet_sample(img_channels, img_rows, img_cols, nb_classes, res_num):
	if res_num == 18:
		return ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
	elif res_num == 32:
		return ResnetBuilder.build_resnet_32((img_channels, img_rows, img_cols), nb_classes)
	elif res_num == 50:
		return ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
	elif res_num == 101:
		return ResnetBuilder.build_resnet_101((img_channels, img_rows, img_cols), nb_classes)
	elif res_num == 152:
		return ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)
	else:
		raise ValueError("There is no suitable Resnet Form.")
