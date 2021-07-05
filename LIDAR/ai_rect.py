#######################################################
#  THIS CODE HAS BEEN OMITTED DUE TO CONFIDENTIALITY  #
#  DO NOT RUN THIS CODE.                              #
#######################################################

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AIPredict:

    def __init__(self):
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=(720, 720, 3)))
        self.model.add(layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="valid"))
        self.model.add(layers.Activation("leakyReLu"))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        ##### FEW MORE CONV2D LAYERS ######

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense())
        self.model.add(layers.Activation("leakyReLu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Dense(classes))
        self.model.add(layers.Activation("softmax"))

        self.model.load_weights(weights.h5)  # NOT REAL NAME

    def predict(self, img):
        prediction = self.model.predict(img)
