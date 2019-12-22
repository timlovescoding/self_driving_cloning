# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of car driving for a few laps 
* Build a convolution neural network that predicts steering angles based off images taken in simulator
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road in autonomous mode 
---


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The convolution neural network that used was derived from a paper written by the team over at Nvidia. For more details of the architecture, please refer to:

[Nvidia Paper](https://arxiv.org/pdf/1704.07911.pdf)

#### 2. Attempts to reduce overfitting in the model

A few dropout layers was included into the architecture to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

No tuning was necessary as an Adam optimizer was used which dynamically adjust the learning rate.


### Model Architecture and Training Strategy

The strategy to tackle this problem was to first split the datasets for training and validation. Next, we have to extract out the inputs and outputs into a format that can be fed into the Convolutional Neural Network. The model architecture was based Nvidia as the Nvidia team has proven that architecture is valid for self driving cars on real roads.

The number epochs was initially set to 10 to figure out at what epoch will the training and validation loss converge/plateau. The loss of the validation loss was initially too high compared to the training. This indicates overfitting. A few dropout layers was added to 
combat overfitting.

During testing when the car is set on autonomous mode (automatically goes around the track), the car does not make very sharp turns which made it fail at one part of the track which a high steering angle was required. This was solved by adding camera images from the left and right angles. Images was also flipped which doubled the amount of training images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Note: Training images used for training are from Udacity. I will extract my own training data for the harder track with more sharp turns and harder manuever in the future.


