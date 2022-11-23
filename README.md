# real-time-face-emotion-detector

# How to run

1. Download the model from https://drive.google.com/file/d/1B_6Phpj36adpaCYwBXHbZLLjk3c9gSSl/view?usp=sharing

2. Run 'live_detection.ipynb' file

- It is possible that some issues arise due to OpenCV VideoCapture. This issues are specific for each type of camera and operating system.

3. To leave the video capture press 'q'.

## Live results

# Information about the project

All of this specifications can be found in the 'model_definition_training.ipynb' file.

## Dataset

The dataset used to train the model was 'fer2013' dataset from kaggle. This dataset consists of black and white images of faces showing emotions.

The emotions covered in the dataset are:

- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

## Model architecture

The model choosen was VGG16:

→ (64, 64, 128, 128, 512, 512) Convolutional layers with 3x3 kernels, padding = 'same' and ReLu activation function

→ (4016, 7) Dense layers, the first also with the ReLu activation function and the last one with softmax.

A batch normalization layer was added after all convolutional and dense layers in order to accelerate convergence and reduce overfitting.

A dropout layer was also added before the last dense layer with proportion of neurons to drop of 0.3. This choice has aldo in mind reducing overfit.

## Data Augmentation

To reduce overfitting, some data augmentation was performed on the training data. The transformations performed include rotation, zoom and horizontal flip.

## Model training

The model was trained for 30 epochs with batches with 32 observations, using Adam optimizer with initial learning rate of 0.0001 and categorical cross entropy losso function.

## Model evaluation and assessment

The model was able to perform predicitons with an accuracy of 0.7 in training and 0.67 in validation.
