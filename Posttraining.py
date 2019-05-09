from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.datasets import cifar10, cifar100
from augmentation import get_cutout_crop
from mixup_generator import MixupGenerator
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.utils.vis_utils import plot_model
from drop_block import DropBlock2D
from SGDR import SGDRScheduler
import numpy as np
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import Model, Input, regularizers, optimizers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training params.
batch_size = 128
epochs = 1023
data_augmentation = True
use_mixup = True

# Load the CIFAR-10 dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
# We assume data format "channels_last".
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

if K.image_data_format() == 'channels_first':
    img_rows = x_train.shape[2]
    img_cols = x_train.shape[3]
    channels = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Basic Block
def basic_block(model):
    model = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model1 = BatchNormalization()(model1)
    model1 = Activation('relu')(model1)
    model = keras.layers.Concatenate()([model1, model])
    model1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model1 = BatchNormalization()(model1)
    model1 = Activation('relu')(model1)
    model2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model2 = BatchNormalization()(model2)
    model2 = Activation('relu')(model2)
    model4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model4 = BatchNormalization()(model4)
    model4 = Activation('relu')(model4)
    model1 = keras.layers.Add()([model4, model1])
    model4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model4 = BatchNormalization()(model4)
    model4 = Activation('relu')(model4)
    model4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model4)
    model4 = BatchNormalization()(model4)
    model4 = Activation('relu')(model4)
    model1 = keras.layers.Concatenate()([model4, model1])
    model7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model7 = BatchNormalization()(model7)
    model7 = Activation('relu')(model7)
    model3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model3 = BatchNormalization()(model3)
    model3 = Activation('relu')(model3)
    model7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model7)
    model7 = BatchNormalization()(model7)
    model7 = Activation('relu')(model7)
    model6 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model7)
    model6 = BatchNormalization()(model6)
    model6 = Activation('relu')(model6)
    model7 = keras.layers.Concatenate()([model6, model7])
    model7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model7)
    model7 = BatchNormalization()(model7)
    model7 = Activation('relu')(model7)
    model1 = keras.layers.Concatenate()([model7, model3])
    model = keras.layers.Concatenate()([model1, model2])
    return model

# Basic Block2
def basic_block2(model):
    model1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model1 = BatchNormalization()(model1)
    model1 = Activation('relu')(model1)
    model2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model2 = BatchNormalization()(model2)
    model2 = Activation('relu')(model2)
    model1 = keras.layers.Add()([model1, model2])
    model2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model2 = BatchNormalization()(model2)
    model2 = Activation('relu')(model2)
    model1 = keras.layers.Add()([model1, model2])
    model3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model3 = BatchNormalization()(model3)
    model3 = Activation('relu')(model3)
    model1 = keras.layers.Concatenate()([model1, model3])
    model2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model2 = BatchNormalization()(model2)
    model2 = Activation('relu')(model2)
    model3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model2)
    model3 = BatchNormalization()(model3)
    model3 = Activation('relu')(model3)
    model5 = keras.layers.Add()([model2, model3])
    model4 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model4 = BatchNormalization()(model4)
    model4 = Activation('relu')(model4)
    model1 = keras.layers.Concatenate()([model5, model4])
    model7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model7 = BatchNormalization()(model7)
    model7 = Activation('relu')(model7)
    model = keras.layers.Concatenate()([model1, model7])
    model = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model1 = BatchNormalization()(model1)
    model1 = Activation('relu')(model1)
    model2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model1)
    model2 = BatchNormalization()(model2)
    model2 = Activation('relu')(model2)
    model1 = keras.layers.Add()([model1, model2])
    model3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model3 = BatchNormalization()(model3)
    model3 = Activation('relu')(model3)
    model4 = keras.layers.Concatenate()([model1, model3])
    model = keras.layers.Add()([model, model4])
    model = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    return model

def model_inil():
    img_input = Input(shape=(32, 32, 3), name='input')
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001), name='first_conv')(img_input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = basic_block(model)
    model = MaxPooling2D(strides=2, name='max_pooling2d_1')(model)
    model = DropBlock2D(block_size=7, keep_prob=0.8, name='drop_block2d_1')(model)
    model = basic_block(model)
    model = MaxPooling2D(strides=2, name='max_pooling2d_2')(model)
    model = DropBlock2D(block_size=5, keep_prob=0.8, name='drop_block2d_2')(model)
    model = basic_block(model)
    model = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001), name='last_conv')(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = GlobalAveragePooling2D()(model)
    model = Dense(10, activation='softmax', name='fc_last')(model)
    model = Model(img_input, model, name='initial_model')
    return model

model = model_inil()

plot_model(model, to_file='architecture1.png', show_shapes=True)

# Instantiate and compile model.
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.1, decay=0., momentum=0.9, nesterov=True), metrics=['accuracy'])
schedule = SGDRScheduler(min_lr=0, max_lr=0.1, steps_per_epoch=np.ceil(x_train.shape[0] / 128), lr_decay=1, cycle_length=1, mult_factor=2)

datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen_test.fit(x_test)
print("Number of parameters: " + str(round(model.count_params() / 1000000, 2)) + "M")

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model_posttraining.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate decaying.
checkpoint = ModelCheckpoint(filepath=filepath,
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint, schedule]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_cutout_crop(crop_shape=[32, 32], padding=4, n_holes=1, length=16))
    datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    datagen.fit(x_train)
    datagen_test.fit(x_test)

    # Fit the model on the batches generated by datagen.flow().
    if use_mixup:
        training_generator = MixupGenerator(x_train, y_train, batch_size=batch_size, alpha=1.0, datagen=datagen)()
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_data=datagen_test.flow(x_test, y_test),
                            epochs=50, verbose=1)
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_data=datagen_test.flow(x_test, y_test),
                            epochs=epochs, verbose=1,
                            callbacks=callbacks)
    else:
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            validation_data=datagen_test.flow(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

# Score trained model.
datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen_test.fit(x_test)

scores = model.evaluate_generator(datagen_test.flow(x_test, y_test), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save('model_posttraining.h5')
