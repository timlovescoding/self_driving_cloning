# **Behavioral Cloning** 

## Project Goals

The goals of this project are the following:
* Use the simulator to collect data of car driving around the track. Steering is done manually using a mouse.
* Build a convolution neural network that predicts steering angles based off images taken in simulator and user steering inputs.
* Train and validate the model with a training and validation set
* Ensure that the model successfully drives around track without leaving the road in autonomous mode 

---


## Model Architecture 


The convolution neural network that used was derived from a paper written by the team over at Nvidia. For more details of the architecture, please refer to:

[Nvidia Paper](https://arxiv.org/pdf/1704.07911.pdf)

A few dropout layers was included into the architecture to reduce overfitting.The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Please view `visualisation.ipynb` file for a detailed summary of the model architecture and sample pictures taken by all of the cameras.


## Training Strategy

The strategy to tackle this problem was to first split the datasets for training and validation. Next, we have to extract out the inputs and outputs into a format that can be fed into the Convolutional Neural Network. The model architecture was based Nvidia as the Nvidia team has proven that architecture performs well even for self driving cars on real roads.

The number epochs was initially set to 10 to figure out at what epoch will the training and validation loss plateau. The loss of the validation loss was initially too high compared to the training. This indicates overfitting. A few dropout layers was added to 
combat overfitting.

During initial testing when the car is set on autonomous mode (automatically goes around the track), the car does not make very sharp turns which made it fail at some parts of the track where a high steering angle was required. This was solved by adding camera images from the left and right angles. An offset correction steering angle was added/subtracted for the left and right camera images.

Training images was also flipped which doubled the amount of training images and also provides better generalization of data.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. Please download `video.mp4` to see the end result.


## How to get started?
---

1. Take the necessary files to drive the car and process images into a video stream:

```
git clone https://github.com/udacity/CarND-Behavioral-Cloning-P3
bash set_git.sh

```

2. Download the Simulator: [**Here**](https://github.com/udacity/self-driving-car-sim)

3. Simulator has two modes, training and autonomous. First, go to training mode to collect data. Press the record button and drive the car around the track yourself.

4. Images and a csv file will be added into your workspace. From the csv file (`driving_log.csv`), it has the path of all of the images and the corresponding steering angles.

5. Head over to my `model.py` to see how I extract the data using information from the `driving_log.csv`. 

6. The fun starts! Train your model and see how it performs on the track (Autonomous Mode)

**ALL THE BEST!** There are tons of resources online in relation to behavioral cloning! Another architecture you can try out is from comma.ai over [**HERE**](https://github.com/commaai/research/blob/master/train_steering_model.py). 


## Advice
---

1. Use an online GPU. With roughly 500k+ parameters, training on your local CPU will take hours and may even heat up your computer. I recommend transferring all of your data into your Google Drive and using Colaboratory to untar+unzip the data, view `visualization.ipynb` to see how I did it.

2. Use a generator. With usage of generator, you won't need to load all of the image arrays in the beginning before you start training. This saves your time especially when you are doing the project over a period of few days as you will not need to re-load the data everytime.  Besides, this is a great practice as you save a lot of memory space using a generator.

3. Crop out unnecessary portion of the image (Example: the sky).

4. Flip the training images (double the dataset) and  move around the track one more time in the opposite direction for better generalization of training data.



