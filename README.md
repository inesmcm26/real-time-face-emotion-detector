# real-time-face-emotion-detector

# How to run

Run 'demo.ipynb' file

- Instead of training the model, you can load the weights of the model that is availale in in the [vgg.h5](/models/vgg.h5) file. There is already one cell that does this.

- It is possible that some issues arise due to OpenCV VideoCapture. This issues are specific for each type of camera and operating system. If you are using a linux system, you can run `v4l2-ctl --list-devices` on the terminal and see which is the number of the camera available. This command will output something like `/dev/video4`, so you know that camera 4 is available. Then, you go to [video_capture.py](/src/video_capture.py) and replace the number in `cv2.VideoCapture(x)` with the number of the camera available.

- To leave the video capture press 'q'.

# Results

TODO

# Information about the project

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

The model choosen was VGG16.

A batch normalization layer was added after all convolutional and dense layers in order to accelerate convergence and reduce overfitting.

A dropout layer was also added before the last dense layer with proportion of neurons to drop of 0.3. This choice has also in mind reducing overfit.

## Data Augmentation

To reduce overfitting, some data augmentation was performed on the training data. The transformations performed include rotation, zoom and horizontal flip.

# Contributions

Contributions are welcomed!

A good first issue would be finding better model parameters to increse the model's accuracy.