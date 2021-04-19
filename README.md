## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./output_images/images.jpg "Images"
[image2]: ./output_images/dist_train.jpg "Training data distrib"
[image3]: ./output_images/dist_valid.jpg "Validation data distrib"
[image4]: ./output_images/dist_test.jpg "Testing data distrib"
[image5]: ./output_images/lenet.jpg "lenet train"
[image6]: ./output_images/incept.jpg "incept train"
[image7]: ./output_images/new_images.jpg "new images"


### Overview

In this project, two models, Lenet and a inception like model, have been used to classify traffic signs. [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) has been used to train and validate the models

The project is done with Tensorflow 2.x and keras. Image augmentation code implemented but not used.
Evaluation of models on a test set shows that an accuracy of 0.9291 and 0.9562 can be achieved with Lenet and Inception models respectively.


### The Project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dataset

The dataset contains about 50000 images of traffic signs, divived into 34799 training samples, 4410 validation images, and 12630 test images.

![alt text][image1]

Training, vaidation, and testing class distributions can be seen in the following pictures:
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Models

Two models are examined here, one being the Lenet model, and the other one is an Incaption like model. The models are defined in model_lenet_fun() and model_incept_fun() functions, respectively. Model.summary() of both models is preseted below:

#### Lenet Model
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 6)         456
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 400)               0
_________________________________________________________________
dense (Dense)                (None, 120)               48120
_________________________________________________________________
dense_1 (Dense)              (None, 84)                10164
_________________________________________________________________
dense_2 (Dense)              (None, 43)                3655
=================================================================
Total params: 64,811
Trainable params: 64,811
Non-trainable params: 0

#### Inception Model

Model: "input_layer"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 32, 32, 3)]  0
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 30, 30, 32)   896         input_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 28, 28, 32)   9248        conv2d_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 14, 14, 32)   0           conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 14, 14, 64)   18496       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 14, 14, 32)   25632       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 14, 14, 16)   528         max_pooling2d_2[0][0]
__________________________________________________________________________________________________
tf.concat (TFOpLambda)          (None, 14, 14, 112)  0           conv2d_4[0][0]
                                                                 conv2d_5[0][0]
                                                                 conv2d_6[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 7, 7, 112)    0           tf.concat[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 7, 7, 32)     32288       max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 7, 7, 16)     44816       max_pooling2d_3[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 7, 7, 8)      904         max_pooling2d_3[0][0]
__________________________________________________________________________________________________
tf.concat_1 (TFOpLambda)        (None, 7, 7, 56)     0           conv2d_7[0][0]
                                                                 conv2d_8[0][0]
                                                                 conv2d_9[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 2744)         0           tf.concat_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 400)          1098000     flatten_1[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 43)           17243       dense_3[0][0]
==================================================================================================
Total params: 1,248,051
Trainable params: 1,248,051
Non-trainable params: 0
__________________________________________________________________________________________________



### Training & Validation

Evaluation of models on a test set shows that an accuracy of 0.9291 and 0.9562 can be achieved with Lenet and Inception models respectively.
Training and validation accuracy for both models are presented below:

![alt text][image5]
![alt text][image6]


### New Images

Below, the prediction results for five new images are shown.

![alt text][image7]




