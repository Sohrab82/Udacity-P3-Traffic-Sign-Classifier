## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./output_images/images.jpg "Images"
[image2]: ./output_images/dist_train.jpg "Training data distrib"
[image3]: ./output_images/dist_valid.jpg "Validation data distrib"
[image4]: ./output_images/dist_test.jpg "Testing data distrib"
[image41]: ./output_images/sample_30.jpg "speed 80"
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

Training, vaidation, and testing class distributions can be seen in the following pictures. As you can see, the number of samples in each class are different than another.

![alt text][image2]
![alt text][image3]
![alt text][image4]

To get an understanding of the images within each class, 25 samples from 80km speed limit class have been plotted below.

![alt text][image41]

### Preprocessing

To normalize the data, X_train, X_valid, and X_test are divided by 255. It was observed that the suggested method of X=(X-128)/128 had an inferior performance compared to division by 255.
```
X_train /= 255
```

Image augmentation was also employed, but showed no improvement over the regular dataset, but also increased the training time considerably, thus commented out.
```
# set up image augmentation
# datagen = ImageDataGenerator(rotation_range=10,
#                             shear_range=0.1,
#                             height_shift_range=0.1,
#                             width_shift_range=0.1,
#                             horizontal_flip=False,
#                             zoom_range=0.2)
# datagen.fit(X_train)
```
Adding the gray scale channel as the forth channel to the images was also examined, but it did not help with aquiring better performance (commanted out in the code).

```
def addGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    image = cv2.merge([b, g, r, gray])
    return image

X_train = np.array(list(map(addGray, X_train)))
```


### Models

Two models are examined here, one being the Lenet model, and the other one is an Incaption like model. The models are defined in model_lenet_fun() and model_incept_fun() functions, respectively. As no overfitting was observed with any of these models, no regularization technique (L2 or dropout or batch-normalization) was used. Model.summary() of both models is preseted below:

#### Lenet Model
```
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
```


#### Inception Model
```
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

```

### Training & Validation

Each model is trained with batch_size=64 and Adam optimizer with learning rate of 0.001 for 30 epochs. After the 30 epochs, not much change was oberverd in training and validation. Larger learning rates failed to train the model as they could not get the model to start learning (training accuracy stuck at < 10%).

```
model_lenet.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history_lenet = model_lenet.fit(X_train, y_train, batch_size=64, epochs=30, shuffle=True, validation_data=(X_valid, y_valid), verbose=1)
```
Evaluation of models on a test set shows that an accuracy of 0.9291 and 0.9562 can be achieved with Lenet and Inception models respectively.
Training and validation accuracy for both models are presented below:

![alt text][image5]
![alt text][image6]


### New Images

Below, the prediction results for seven new images are shown. The images are downloaded from internet and in some cases noise or background colors have been added. 
It can be seen that in the fifth image, the background of the image has resulted in a wrong prediction (`speed limit 30km/h`) while the second guess is the correct prediction (`Road work`).

Also, in the last two images the prediction (`speed limit 30km/h`) is wrong which can be linked to having too many samples of `speed limit 30km/h` in the training data compared to `Road narrows on the right` and other classes.

![alt text][image7]




