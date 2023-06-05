import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os
import cv2

import numpy as np
from matplotlib import pyplot as plt


class LogoDetection:

    def __init__(self):

        self.input_size = (108, 192)
        self.channels = 3
        self.color = 'rgb'

        self.train_data = tf.data.Dataset
        self.validation_data = tf.data.Dataset
        self.test_data = tf.data.Dataset

        self.model = Sequential()
        self.classifier = Sequential()

    def generate_data(self):
        train_data = tf.keras.utils.image_dataset_from_directory('./images/train', image_size=self.input_size,
                                                                 batch_size=32, labels='inferred', label_mode='binary',
                                                                 color_mode=self.color, shuffle=True)
        train_data = train_data.map(lambda x, y: (x / 255, y))

        validation_data = tf.keras.utils.image_dataset_from_directory('./images/validation', image_size=self.input_size,
                                                                      batch_size=32, labels='inferred', label_mode='binary',
                                                                      color_mode=self.color, shuffle=True)
        validation_data = validation_data.map(lambda x, y: (x / 255, y))

        test_data = tf.keras.utils.image_dataset_from_directory('./images/test', image_size=self.input_size,
                                                                batch_size=32, labels='inferred', label_mode='binary',
                                                                color_mode=self.color, shuffle=True)
        test_data = test_data.map(lambda x, y: (x / 255, y))

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def create_model(self):
        classifier = Sequential()

        classifier.add(
            Conv2D(32, (3, 3), input_shape=(*self.input_size, self.channels),
                   activation='relu'))  # Convolutional layer.
        classifier.add(MaxPooling2D(pool_size=(2, 2)))  # Down-sampling.
        classifier.add(Conv2D(32, (3, 3), input_shape=(31, 31, 32), activation='relu'))  # Convolutional layer.
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())  # Projecting to lower dimensions.

        # Construct input layer with 128 inputs, regularize to prevent overfitting in between and output a binary
        # predictive outcome(cats and dogs)
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=1, activation='sigmoid'))

        # Compiling our CNN.
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.summary()

        model = classifier.fit(self.train_data, epochs=20, validation_data=self.validation_data,
                               validation_steps=10)

        self.model = model
        print('Model is now: ' + str(model))

        self.classifier = classifier
        print('Classfier is now: ' + str(classifier))

    def plotting_loss(self):
        fig = plt.figure()
        plt.plot(self.model.history['loss'], color='teal', label='loss')
        plt.plot(self.model.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig = plt.figure()
        plt.plot(self.model.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.model.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

    def save_model(self):
        self.classifier.save(os.path.join('models', 'logo-detection-simple.h5'))

