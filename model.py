import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil
#from matplotlib import pyplot as plt


samples = []

with open('./data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)    
    next(reader) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)

        

# Input and output has to be split for training and testing.
# in the training set (later in Keras, it will split into validation set too)
train_samples,validation_samples = train_test_split(samples, test_size=0.30)

def generator(samples , batch_size = 64): # generate a single batch per execution
# With this function, we do not need to load all the images and wait. It will only load when it is needed.

    sample_size = len(samples) 

    while True:
        #print('An epoch to pass')
        for i in range(0,sample_size,batch_size):
            
            batch_samples = samples[i:i+batch_size]
            
            images = []
            angles = []
          
            for batch_sample in batch_samples:
                
                name_center = './data/IMG/'+ batch_sample[0].split('/')[-1]
                name_left   = './data/IMG/'+ batch_sample[1].split('/')[-1]
                name_right  = './data/IMG/'+ batch_sample[2].split('/')[-1]


                
                #drivepy takes in RGB pictures (model has to be trained in RGB format
                center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(name_left), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2RGB)


                center_angle = float(batch_sample[3])

                # Right angle > is positive angle
                # left angle < is negative angle
                # Try: 0.7 reactive and able to deal with sharp turns but very wobbly for small angles
                correction  = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction 
                
                #print(center_image)
                images.extend([center_image,left_image,right_image])
                angles.extend([center_angle,left_angle,right_angle])
                
            
            inputs = np.array(images)
            outputs = np.array(angles)
            
        
            #Extra: inputs and outputs ( Idea: Flip image to add data + variety )
            
            extra_input = []
            extra_output = []
            
            for x,y in zip(inputs,outputs):
                
                extra_input.append(x)
                new_x =  np.fliplr(x)
                extra_input.append(new_x)
                
                extra_output.append(y)
                new_y = y*-1.0  # reverse steering for the flipped image
                extra_output.append(new_y)
            
            extra_input = np.array(extra_input)
            extra_output = np.array(extra_output)
            
            
            # preporcess here
            #print('Sent a single batch')
            #print(extra_input.shape)
            #print(extra_output.shape)
            yield shuffle(extra_input, extra_output)
         
   

#####****** Model Set up********************************


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense , Cropping2D
from keras.layers import Flatten, Dropout , BatchNormalization , Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D

def NvidiaModel():
    
  model = Sequential()
  
  model.add(Lambda(lambda x: (x/255.0)-0.5 , input_shape = (160,320,3)))
  model.add(Cropping2D(cropping=( (70, 20), (0, 0) )))

  model.add(Conv2D(24,  (5,5), strides=(2, 2),activation = 'relu'))
  model.add(Conv2D(36,  (5,5), strides=(2, 2), activation = 'relu'))
  model.add(Conv2D(48,  (5,5), strides=(2, 2),activation = 'relu'))

  model.add(Conv2D(64,  (3,3), activation = 'relu'))
  model.add(Conv2D(64,  (3,3), activation = 'relu'))

  #model.add(BatchNormalization())
  model.add(Dropout(0.5))
  

  model.add(Flatten())
  model.add(Dense(100,activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation = 'relu'))
  model.add(Dense(10, activation = 'relu'))
  model.add(Dense(1))

  model.compile(optimizer = 'adam', loss='mse')
  #Adam(lr = 0.02),

  return model


model = NvidiaModel()
model.summary()

batch_size = 32
training_generator = generator(train_samples,batch_size)
validation_generator = generator(validation_samples,batch_size)
#fit_generator takes in a generator of tuple(input,output) to fit the data

# Reminder: Training samples is doubled due to the extra flipped images! And then tripled due to 3 camera angles. 
# Therefore, each batch that is being sent to the generator (being yield) is 6 times the size.
# Example: if you set batch_size = 10, you are actually doing a batch of 60



history_object = model.fit_generator(training_generator, 
                    steps_per_epoch=ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=ceil(len(validation_samples)/batch_size), 
                    epochs= 5 , verbose=1)


model.save('model.h5')


