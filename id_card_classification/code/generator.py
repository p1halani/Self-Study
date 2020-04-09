import keras
import cv2
import os
import math
import numpy as np

class DataGenerator(keras.utils.Sequence):

        def __init__(self, list_IDs, labels, image_path,
                    to_fit=True, batch_size=32, dim=(192, 192),
                    n_channels=3, n_classes=10, shuffle=False):
            self.list_IDs = list_IDs
            self.labels = labels
            self.image_path = image_path
            self.n_channels = n_channels
            self.to_fit = to_fit
            self.n_classes = n_classes
            self.dim = dim
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.n = 0
            self.on_epoch_end()

        def __next__(self):
            # Get one batch of data
            data = self.__getitem__(self.n)
            # Batch index
            self.n += 1

            # If we have processed the entire dataset then
            if self.n >= self.__len__():
                self.on_epoch_end
                self.n = 0

            return data
        
        def __len__(self):
            # Return the number of batches of the dataset
            return math.ceil(len(self.indexes)/self.batch_size)
        
        def __getitem__(self, index):
        
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X = self._generate_x(list_IDs_temp)

            if self.to_fit:
                y = self._generate_y(indexes)
                return X, y
            else:
                return X
        
        def on_epoch_end(self):
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
            
        def _generate_x(self, list_IDs_temp):
            # Initialization
            X = np.empty((self.batch_size, *self.dim, 1))

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = self._load_blue_channel_image(os.path.join(self.image_path, ID))

            return X
        
        def _generate_y(self, indexes):

            y = np.empty((self.batch_size, 1), dtype=int)

            # Generate labels
            for i,idx in enumerate(indexes):
                # Store sample
                temp = self.labels[idx]
                if temp.strip() == '2':
                    z = 2
                if temp.strip() == '1':
                    z = 1
                if temp.strip() == '0':
                    z = 0
                y[i,] = z

            return keras.utils.to_categorical(y, num_classes = self.n_classes)
        
        def _load_blue_channel_image(self, image_path):
            img = cv2.imread(image_path)
            img = img[:,:,0].reshape(192,192,1)
            img = img / 255
            return img