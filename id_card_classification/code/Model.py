import keras
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class CNNModel():
    def __init__(self, params):
        self.model = Sequential()
        for i in range(len(params['filters'])):
            if(i==0):
                self.model.add(Conv2D(params['filters'][i], (3, 3), activation='relu', input_shape=params['input_shape']))
            else:
                self.model.add(Conv2D(params['filters'][i], (3, 3), activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        for i in range(len(params['dense'])):
            self.model.add(Dense(params['dense'][i], activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.5))

        self.model.add(Dense(params['n_classes'], activation='softmax'))

    def getModel(self):
        return self.model