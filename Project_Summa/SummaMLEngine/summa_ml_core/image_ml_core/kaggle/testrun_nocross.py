import os
import time
import pickle
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import LabelShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


from keras.preprocessing.image import ImageDataGenerator

from model import Model
from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp
from architectures import vgg_16, resnet_sample

from keras.optimizers import SGD, RMSprop, Adagrad, Adam

img_channels = 3
img_rows = 64
img_cols = 64
nb_classes = 10

#DATASET_PATH = 'dataset/data_5.pkl'
DATASET_PATH = 'dataset/data_'+str(img_rows)+'x'+str(img_cols)+'.pkl'
CHECKPOINT_PATH = 'checkpoints/'
SUMMARY_PATH = 'summaries/'

NUM_EPOCHS = 40
MAX_FOLDS = 8
# for vgg_16, upto 16*16
# for resnet, downto 32*32
BATCH_SIZE = 16*16
#BATCH_SIZE = 32*32

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)

X_train_raw.astype('float32')
X_test.astype('float32')


X_train_raw = X_train_raw/255
X_test = X_test/255
#X_train_raw /= 255
#X_test /= 255
# error ! : TypeError: ufunc 'true_divide' output (typecode 'd') could not be coerced to provided output parameter (typecode 'l') according to the casting rule ''same_kind'' 

#np.divide(X_train_raw, 255, out=X_train_raw)
#np.divide(X_test, 255, out=X_tset)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw,test_size=0.3, random_state=42)

print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape)


print('Image Data Normalization')
datagen = ImageDataGenerator(
	featurewise_center=True,
	samplewise_center=True,
	featurewise_std_normalization=True,
	samplewise_std_normalization=True,
	zca_whitening=False,
	rotation_range=0,
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=False,
	vertical_flip=False
	)

datagen.fit(X_train)


# 18, 50 
model = resnet_sample(img_channels, img_rows, img_cols, nb_classes, 18)
#model = vgg_16()

print('model is set..')
print(model)

# Optimizers
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#adagrad = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),\
	validation_data=(X_valid, y_valid),\
	samples_per_epoch=X_train.shape[0],\
	nb_epoch=NUM_EPOCHS, verbose=1)


print('Begin evaluation...')
score = model.evaluate(X_valid, y_valid, batch_size=BATCH_SIZE, verbose=1)
predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

print(score)

print('score: {}...'.format(score))
submission_dest = os.path.join(SUMMARY_PATH, 'submission_{}_{}.csv'.format(int(time.time()), score))
write_submission(predictions, X_test_ids, submission_dest)


print('Done.')


