import os

import cv2
import keras.utils.metrics_utils
import numpy
from matplotlib import pyplot as plt
from keras.models import load_model
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import re


class Predict:

    def __init__(self):
        self.path = 'images/prediction_set/'
        self.input_size = (108, 192)
        self.classes = ['game', 'logo']
        self.color = 'rgb'

        self.logo_list = []
        self.sequence_list = []
        self.startframes = []

    # Load model, preprocess and predict image.
    def predict_images(self, filename):
        loaded_model = load_model('./models/logo-detection-simple.h5')
        logo_list = []
        for i in os.listdir(self.path + filename + '/'):
            img = cv2.imread(self.path + filename + '/' + i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resize = tf.image.resize(img, self.input_size)

            lynhat = loaded_model.predict(np.expand_dims(resize / 255, 0))

            if lynhat > 0.5:
                logo_list.append(i)

        self.logo_list = logo_list

    def find_seq_v2(self):
        sorted_list = sorted(self.logo_list)
        digits = []
        for string in sorted_list:
            string_nr = ''
            for char in string:
                if char.isdigit():
                    string_nr += char
            digits.append(int(string_nr))

        stretches = []
        i = 0
        while i < len(digits):
            j = i
            while j < len(digits) - 1 and digits[j + 1] - digits[j] == 1:
                j += 1
            if j - i + 1 >= 11:
                stretches.append(digits[i:j + 1])
            i = j + 1
        self.sequence_list = stretches

    def handle_stretches(self, stretch):
        logo_startframes = []
        for i in stretch:
            logo_startframes.append(i[0])
        self.startframes = sorted(logo_startframes)

    def show_pics(self):
        sequence_list = self.sequence_list

        # PLOTTING
        for i in sequence_list:
            for count in range(-3, 4):
                frame_nr = i + count
                img = cv2.imread('./images/prediction_set/79588/frame' + str(frame_nr) + '.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title('Frame nr: ' + str(frame_nr))
                plt.show()

    def create_confusion_matrix(self):
        loaded_model = load_model('./models/logo-detection-simple-72x72x3.h5')

        test_data = tf.keras.utils.image_dataset_from_directory('./images/test',
                                                                image_size=self.input_size,
                                                                batch_size=32, labels='inferred', label_mode='binary',
                                                                color_mode=self.color, shuffle=False)

        test_data = test_data.map(lambda x, y: (x / 255, y))

        y_true = []
        y_pred = []
        for images, labels in test_data:
            predictions = loaded_model.predict(images)
            y_true += labels.numpy().tolist()
            y_pred += (predictions > 0.5).astype(int).tolist()

        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        print(conf_matrix)

        classes = ['game', 'logo']

        # Plot the conf_matrix
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

