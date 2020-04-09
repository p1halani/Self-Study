from keras.preprocessing.image import load_img, array_to_img, img_to_array
import os
import numpy as np
import pandas as pd
import keras
import argparse
import cv2
from keras.models import Sequential, model_from_json

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required = True)
args = vars(parser.parse_args())

json_file = open('../artifacts/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../artifacts/model.h5")


img_path = args["path"]
img = cv2.imread(img_path)
img = img[:,:,0].reshape(192,192,1)
img = img / 255
# img = load_img(img_path, target_size=(192, 192, 3))
# print(img.size)
# img = img_to_array(img)[:,:,0]                    # (height, width, channels)
# print(img.size)
# var = input()
img = np.expand_dims(img, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
# img /= 255


predictions = model.predict_classes(img)

print(predictions)