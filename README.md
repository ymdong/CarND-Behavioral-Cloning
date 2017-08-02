#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/image_flip.png "Flipped Image"
[image3]: ./images/center.jpg "Center Camera Image"
[image4]: ./images/left.jpg "Left Camera Image"
[image5]: ./images/right.jpg "Recovery Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
      python drive.py model.h5 
```
Besides, in model.py, cv2 is used to read the image in BGR format, to be consistent with this format, we add one line in drive.py to convert the RGB to BGR format after reading the image:
```sh
      imgString = data["image"]
      image = Image.open(BytesIO(base64.b64decode(imgString)))
      image_array = np.asarray(image)
      image_array = image_array[:,:,::-1]  # this line converts image from RGB to BGR
      steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model and it is clearly organized and commented.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

NVIDIA CNN architecture is adopted in this project, which includes 5 convolutional layers and 4 fully-connected layers. The model architecture is complex enough to capture the required features in this autonomous driving task. 

####2. Attempts to reduce overfitting in the model

Train/validation splits have been used to avoid overfitting and the epochs number is reduced accordingly when overfitting occurs. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

The training data is collected by carefully driving the car on track one within the lane for 3 laps. The multiple camera data is used to help the car recover from the left and right of the road. The steering angle is corrected by experiment for the left and right camera image data. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Firstly, I preprocess the collected data. I combine the data of center, left and right camera images and correct the steering angle for left and right camera images. Then I augment the data set by flipping the image, which can help the model to better generalize the model. Before feeding the data into CNN model, I also preprocess the data by normalization, mean-centering and cropping the image. 

Secondly, I adopt a existing NVIDIA CNN model introduced in the class, and this model is good starting point because it is already well trained and complex enough to capture the required features. 

Thirdly, I run the model on AWS GPU for 4 epochs, and it turns out the training loss is decreasing but the validation loss increases after 2 epochs, which implies the model is overfitting. Thus, I reduce the epoch number to 2.

Finally, when I run the trained model in the simulator, I find the car will leave the road in the second sharp turn. So I adjust the steering angel correction of the left and right image from 0.2 to 0.1. Then the car can autonomously drive around track one without leaving the road, which is recorded in the video file. 

####2. Final Model Architecture

As described previously, I use the NVIDIA’s CNN model. Below is a visualization of this architecture using Keras model visualization tool, we can find after the input layer, there is Lambda layer to normalize and mean-center the data, then Cropping2D layer to crop out the irrelevant part of image, then there are 5 convolutional layers, finally 4 fully-connected layers come after a flatten layer. 

![alt text][image1]

####3. Creation of the Training Set & Training Process

As described previously, I first collect the data by doing center-lane driving on track one for 3 laps. Here is an example of center lane driving.

![alt text][image3]

Then the left and right camera images are also used to help the car staying in the center and their steering angles are corrected. Here are examples of left and right camera images corresponding to above center camera image:

![alt text][image4]
![alt text][image5]

To augment the dataset for better generalizing the model, I also flipped images and here are examples of original image and flipped image:

![alt text][image3]
![alt text][image2]

After the collection process, I have 25740 data points, then I preprocess the data by normalization, mean centering and cropping. I finally randomly shuffled the data set and put 20% of the data into a validation set. An adam optimizer is used without the need of manually setting learning rate. The number of epochs is set as 2 by observing the training and validation loss. 
