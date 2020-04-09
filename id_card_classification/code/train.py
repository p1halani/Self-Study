import click as ck
import os
import numpy as np
import pandas as pd
import cv2
import logging

import keras
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.models import load_model
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split

from Model import CNNModel
from generator import DataGenerator

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--labels', '-lb', default='../data/gicsd_labels.csv',
    help='Path to the data directory')
@ck.option(
    '--images-dir', '-dir_img', default='../data/images',
    help='Path to the data directory')
@ck.option(
    '--model-file', '-mf', default='../artifacts/model.h5',
    help='Model')
@ck.option(
    '--out-file', '-of', default='../artifacts/model.json',
    help='Model')
@ck.option(
    '--weights-file', '-wf', default='../artifacts/model.h5',
    help='Model')
@ck.option(
    '--split', '-s', default=0.2,
    help='train/test split')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=150,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='../artifacts/training.csv',
    help='Batch size')

def main(logger_file, load, epochs, batch_size, split, model_file,
            images_dir, labels, weights_file, out_file):

    params = {
        'filters' : [32, 64, 128],
        'dense' : [512],
        'loss' : 'categorical_crossentropy',
        'optimizer' : Adam(lr=3e-4),
        'input_shape' : (192,192,1),
        'n_classes' : 3
    }
    print('Params:', params)

    data_idx, labels_idx = load_data(images_dir=images_dir, labels=labels)

    X_train, X_test, y_train, y_test = train_test_split(data_idx, labels_idx, test_size=split, random_state=42)

    train_generator = DataGenerator(X_train, y_train,image_path = images_dir, n_classes = 3, batch_size=batch_size)
    val_generator = DataGenerator(X_test, y_test,image_path = images_dir, n_classes = 3, batch_size=batch_size)

    if load:
        logging.info('Loading pretrained model')
        model = load_model(model_file)
    else:
        logging.info('Creating a new model')
        model = CNNModel(params)
        model = model.getModel()
        model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=['accuracy'])
        logging.info('Compilation finished')
        
        logging.info("Training data size: %d" % len(X_train))
        logging.info("Validation data size: %d" % len(X_test))
        checkpointer = ModelCheckpoint(
                    filepath=model_file,
                    verbose=1, save_best_only=True)
        earlystop = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
        logger = CSVLogger(logger_file)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                    patience=2, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001)
        callbacks = [earlystop, learning_rate_reduction]

        model.summary()

        logging.info('Starting training the model')


        train_steps = len(train_generator)
        valid_steps = len(val_generator)
        history = model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_generator,
            validation_steps=valid_steps,
            callbacks=callbacks,
            workers=12
        )

        model_json = model.to_json()
        with open(out_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(weights_file)

def load_data(images_dir, labels):
    data_ids = [path for path in os.listdir(images_dir)]
    data_df = pd.read_csv(labels)
    data_df.columns = ['IMAGE_FILENAME', 'LABEL']
    def rule(x):
        if x['LABEL'].strip() == 'NO_VISIBILITY':
            return '2'
        if x['LABEL'].strip() == 'PARTIAL_VISIBILITY':
            return '1'
        if x['LABEL'].strip() == 'FULL_VISIBILITY':
            return '0'
        
        return np.nan

    data_df['LABEL'] = data_df.apply(rule, axis = 1)

    data_idx = data_df['IMAGE_FILENAME'].tolist()
    labels_idx = data_df['LABEL'].tolist()
    return (data_idx, labels_idx)


if __name__ == '__main__':
    main()