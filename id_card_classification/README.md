***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 


## Dependencies

Pandas Version: 0.25.1
Numpy Version: 1.17.2
OpenCV Version: 4.1.0
Seaborn Version: 0.9.0
Keras Version: 2.2.4
Matplotlib: 3.1.1
Python: 3.6.9
Scikit-learn: 0.21.3
[TODO: Complete this section with the main dependencies and how to install them]

## Run Instructions

python3 train.py

python3 predict.py

If output == 2:
	No Visibility
If output == 1:
	Partial Visibility
If output == 0:
	Full Visibility
[TODO: Complete this section with how to run the project]

## Approach

After exploring the data, I came to know that only blue channel is contributing to features. Hence used blue channel of every image to train a CNN model with Adam optimizer. Number of classes are 3. Hence softmax is used with 3 neurons in last layer, while other is having a block of (Convolution layer, BatchNormalization, Maxpooling, Dropout).
The callbacks like Earlystopping, ReduceLRonPlateau are used.

Note : I haven't completed the training though attached a screenshot till 13 epochs in artifacts folder.
[TODO: Complete this section with a brief summary of the approach]

## Future Work

Will use Transfer Learning.
SMOTE for oversampling as dataset is highly biased.
Will break the code in different modules.
Use K-Fold Cross-Validation.
Plot accuracy and loss curve.
I think that BACKGROUND_ID may be helpful which is present in image name.
[TODO: Complete this section with a set of ideas for future work]