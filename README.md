#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[imageSteeringC]: ./examples/steeringCenterCam.png "Steering Center Camera"
[imageSteeringThreeCam]: ./examples/steeringThreeCam.png "Steering Three Cameras"
[imageModel]: ./examples/model.png "Model Architecture"
[imageLoss]: ./examples/loss.png "Training Validation Loss"
[imageCenter]: ./examples/center_2017_08_29_22_27_01_906.jpg "Center Camera"
[imageLeft]: ./examples/left_2017_08_29_22_27_01_906.jpg "Left Camera"
[imageRight]: ./examples/right_2017_08_29_22_27_01_906.jpg "Right Camera"

[imageOrig]: ./examples/imgOrig.jpg "Origina Image"
[imageFlip]: ./examples/imgFlip.jpg "Flipped Image"
[imageCV2]: ./examples/imgCV2.jpg "BGR"
[imageLoss]: ./examples/loss.png "Loss vs epoch"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

I submit a zip file that includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 shows the video taken during autonomous mode
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model begins with a lambda layer that normalizes the input images (model.py line 42). Then the image is cropped by 70 pixels on the top, and 25 pixels on the bottom (model.py line 43). The cropping removes the area with information that is not relevant to the goal.

Next step is a convolutional layer with a filter size 5x5, stride 3, and depth 12 (model.py line 44). The activation was set to RELU, rectified linear unit, to introduce nonlinearity, and to avoid vanishing gradient. The convolution layer was followed by a maxpooling layer of 2x2 (model.py line 45). Then the same convolution/maxpooling layers were repeated once.

Lastly, a fully connected layer with 160 nodes was added with a dropout rate of 0.5 and RELU activation, followed by one output node (model.py lines 49-51).

####2. Attempts to reduce overfitting in the model


The fully connected layer has dropout rate of 50% (model.py line 50).

The data was randomly divided and 20% of the data was used for validation (model.py line 54).

The trained model was tested with the simulator to ensure that the car stays in the lane.

####3. Model parameter tuning

Over a hundred trains were performed to determine the parameters for the convulutional layers and the fully connected layer.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 53).

####4. Appropriate training data

I have collected the data by driving the vehicle for two laps, then driving one lap in the opposite direction. The images taken by three cameras were used.

The images were also randomly flipped to form mirror images (model.py line 27).

###Architeccture and Training Documentation

####1. Solution Design Approach

So far in the previous projects, the combination of convolutional layers and fully connected layers performed well dealing with image data. So I started with a similar setup, and ran over a hundred trainings with varying architectures and parameters to determine the best combination.

My first model was trained on the images from the center camera only. After a few adjustments, I was able to achieve very low loss values, both for training and validation. I thought it would be a very good model. However, in the simulation, the steering values stayed around zero and the car failed to recover at the very first curve. It seemed that the steering values from the center camera clustered around zero, and the network was learning to keep the steering around zero. That would not result in a successful driving in the simulator, but the loss function could have been made very small. Below is the histogram of the steering values from the center camera images of the training data.

![alt text][imageSteeringC]

With this kind of input distribution, it seems unlikely that the network would learn to recover in case the car gets too close to either side. After including the images from the left and right cameras, as I will discuss more in detail in a later section, the distribution looked like this.

![alt_text][imageSteeringThreeCam]

The model architecture and the parametes were tuned by running the training over a hundred times with varying settings, and comparing the validation loss values. The dropout of 50% was used in the fully connected layer to reduce the overfitting. The final model architecture is shown in the next section.

Although I was fairly confident that the network was doing a good job achieving the minimal loss function, the simulation showed that the car was not recovering in some curves. I thgough it would help if I augment the data set to feed more information to the network. I thought about a few ways to achieve that goal, and I decided to flip some images to form mirror images. Specifically, I tried flipping every other images, thinking that any two consecutive images from the simulation would be very similar to each other, and therefore may be redundant. I have added some randomness to the order, rather than strictly taking turns with flipping and not flipping, to avoid any systematic effect from the regularness.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes.

| Layer                 |     Description                                    | 
|:---------------------:|:--------------------------------------------------:| 
| Input                 | 160x320x3 RGB image                                | 
| Lambda                | Normalization                                      |
| Cropping              | outputs 65x320x3                                   |
| Convolution 5x5       | RELU, 3x3 stride, valid padding, outputs 21x106x12 |
| Max pooling           | 2x2 stride,  outputs 10x53x12                      |
| Convolution 5x5       | RELU, 3x3 stride, valid padding, outputs 2x17x12   |
| Max pooling           | 2x2 stride,  outputs 1x8x12                        |
| Fully connected       | 160 nodes, RELU, 50% dropout                       |
| Output                | 1 node                                             |


####3. Creation of the Training Set & Training Process

At first, I drove the simulator using the keyboard. However, I was concerned of the fact that I was able to turn a curve by hitting left or right arrow only intermittently. As I lift my finger off the arrow keys, the steering turned back to zero. That would create a lot of data points with zero steering along the curve, which is not the cars are usually driven. So I swithced to using mouse to steer.

Using mouse was not easy because the pointer was moving too fast even with the system setting at the slowest pointer speed. I use a Linux machine, and I was able to find a utility called 'xinput' that changes input device settings. With the program, I was able to set the pointer speed much slower, making it possible to steer the car very smoothly.

I have recored the driving for two laps. Then I made a U-turn to drive the car in the opposite direction, and recorded another lap. I have trained a network for a test using only images from the center camera. After a few trials with different architectures, I was able to achieve very low validation loss. However, when I ran the simulation with the model, the car did not stay between the lanes. It seemed that the network learned to keep the steering near zero. I thought that by using the images only from the center camera, I was feeding a lot of data points with a small steering value. It seemed that I needed to add more date with 'recovery laps', as suggested in the lesson material.

I started recording recovery moves. With the recording turned off, I drove the car close to a lane, turned recording on, then steered the car to the center of the road. This was rather a tedious job, but adding those data seemed to improve the performance somewhat, so I was encouraged. However, I thought including the images from the left and right cameras would have similar effect as doing recovery laps, so I moved on to that idea instead. Here are an example of the images from center, left, and right cameras.


![alt text][imageCenter] ![alt text][imageLeft] ![alt text][imageRight]

The steering values for the two side cameras needed to be adjusted. I have tried 21 different values between 0.1 and 0.3. There was obvious over-correction with the value 0.3, leading to severe zig-zagging. With the value set to 0.1, the network seemded to undertrain and the car was not recovering quick enough. The ideal value seemed to be 0.13. Including the images from left and right cameras improved the performance a lot.

20% of the data was randomly selected and used for validation sample. The data was shuffled before training. Adam optimizer was used for fast training and to avoid tuning the learning rate. I was able to train a network that satisfies the project requirements pretty soon.

I almost wanted to wrap up the project with the successful result. However, when I trained the network once again with the same setting, the test run was not successful. It seemed that the result was not robust. I decided to put in more work to improve the model. So I tried flipping randomly selected images. Follwing images show the effect of flipping. The original image is on the left, taken from the above center camera image. On the right hand side is the flipped image.

![alt text][imageOrig] ![alt text][imageFlip]

The flipped images seemed to have an effect of generalizing the train sample. The car in the simulation stayed more stable. Then I noticed a problem in the code.

I was using the function cv2.imread() to read in the image files for training. While I was visualizing some examples, I noticed that the color of the images that I read with the script did not look right. Here is an example.

![alt_text][imageCV2]

After some investigation, I learned that the function cv2.imread() loads the color information in BGR, rather than in RGB. It could be a problem if the simulator feeds the color field in RGB when the model is trained with the color data ordered in BGR. To see if this is the case, I looked into the file drive.py. The image from the simulator was getting converted to an array by following function call (drive.py line 62).

`image = Image.open(BytesIO(base64.b64decode(imgString)))`

Since the function Image.open() belongs to PIL package the color would be in RGB, rather than BGR. To see how much problem it was causing, I trained two sets of ten models. One set is trained with the data read by the function cv2.imread() which loads the color in BGR. The other set is trained with the function matplotlib.pyplot.imread(), which loads the color in RGB. I have set the speed to 30, rather than the default value 9 (drive.py line 47). This would have an effect of amplifying the instability, making it easier to see the differences between two sets of models.

When I tested the first set of models, which are trained with the BGR data, only two of ten models were able to complete the lap. On the other hand, the second group completed the lap nine times out of ten. I think it is a remarkable manifest of stability, given that the high speed that I was using. Also it shows the importance of the color information for producing the correct prediction. This table summarizes the test.

| Training color | Simulator color | nTrial | Success         | 
|:--------------:|:---------------:|:------:|:---------------:| 
| *BGR*          | RGB             | 10     | 2               |
| RGB            | RGB             | 10     | 9               |


I have set the number of epochs to 10. Following plot shows that the validation loss is almost flat after 6th epoch.

![alt_text][imageLoss]

The video (video.mp4) was created with the speed set to 30 in the file drive.py. Even though there is some zig-zagging, the car stays between the lane lines and satisfies the project requirements.
