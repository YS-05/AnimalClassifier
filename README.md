# AnimalClassifier

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Conclusion](#conclusion)
- [Further Improvements](#further-improvements)
- [License](#license)

## Introduction

This project demonstrates an end-to-end image classification task using 2 different Convolutional Neural Networks (CNN) models. The goal of this project is to classify 10 different animals based on images. It includes two phases:

A baseline CNN model that we developed entirely from scratch.
A more advanced transfer learning approach using a pre-trained VGG16 model fine-tuned for our dataset.

## Dataset

Initially, the dataset consisted of around 28000 images; however, to keep the same number of images for each animal and eliminate bias, we ended up actually using 11560 images for training, 1450 images for validation, and 1450 for testing. Hence, the dataset was broken down into 80-10-10 ratios.

The dataset used contains images of the following animals:

Cat
Dog
Horse
Cow
Elephant
Squirrel
Butterfly
Chicken
Sheep
Spider

You can download the dataset on Kaggle through this [link](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

## Model Architectures

The baseline CNN model consists of several convolutional layers for feature extraction, followed by max-pooling layers to reduce dimensionality, and finally, fully connected (dense) layers for classification:
```plaintext
Model: "sequential"
Layer (type)                    Output Shape                Param #   
==================================================================
conv2d (Conv2D)                 (None, 148, 148, 64)        1,792    
max_pooling2d (MaxPooling2D)    (None, 74, 74, 64)          0        
conv2d_1 (Conv2D)               (None, 72, 72, 128)         73,856   
max_pooling2d_1 (MaxPooling2D)  (None, 36, 36, 128)         0        
conv2d_2 (Conv2D)               (None, 34, 34, 128)         147,584  
max_pooling2d_2 (MaxPooling2D)  (None, 17, 17, 128)         0        
conv2d_3 (Conv2D)               (None, 15, 15, 256)         295,168  
max_pooling2d_3 (MaxPooling2D)  (None, 7, 7, 256)           0        
flatten (Flatten)               (None, 12544)               0        
dense (Dense)                   (None, 512)                 6,423,040
dropout (Dropout)               (None, 512)                 0        
dense_1 (Dense)                 (None, 10)                  5,130    
==================================================================
Total params: 6,946,570
Trainable params: 6,946,570
Non-trainable params: 0
```

VGG16 Transfer Learning Model
The VGG16 model, pre-trained on ImageNet, is modified by adding a global average pooling layer and two dense layers for classification into the 10 animal classes. Here is the modified architecture:
```plaintext
Model: "functional"
Layer (type)                         Output Shape                Param #   
==================================================================
input_1 (InputLayer)                 [(None, 224, 224, 3)]       0         
block1_conv1 (Conv2D)                (None, 224, 224, 64)        1,792     
block1_conv2 (Conv2D)                (None, 224, 224, 64)        36,928    
block1_pool (MaxPooling2D)           (None, 112, 112, 64)        0         
block2_conv1 (Conv2D)                (None, 112, 112, 128)       73,856    
block2_conv2 (Conv2D)                (None, 112, 112, 128)       147,584   
block2_pool (MaxPooling2D)           (None, 56, 56, 128)         0         
block3_conv1 (Conv2D)                (None, 56, 56, 256)         295,168   
block3_conv2 (Conv2D)                (None, 56, 56, 256)         590,080   
block3_conv3 (Conv2D)                (None, 56, 56, 256)         590,080   
block3_pool (MaxPooling2D)           (None, 28, 28, 256)         0         
block4_conv1 (Conv2D)                (None, 28, 28, 512)         1,180,160 
block4_conv2 (Conv2D)                (None, 28, 28, 512)         2,359,808 
block4_conv3 (Conv2D)                (None, 28, 28, 512)         2,359,808 
block4_pool (MaxPooling2D)           (None, 14, 14, 512)         0         
block5_conv1 (Conv2D)                (None, 14, 14, 512)         2,359,808 
block5_conv2 (Conv2D)                (None, 14, 14, 512)         2,359,808 
block5_conv3 (Conv2D)                (None, 14, 14, 512)         2,359,808 
block5_pool (MaxPooling2D)           (None, 7, 7, 512)           0         
global_average_pooling2d (GlobalAvg) (None, 512)                 0         
dense (Dense)                        (None, 512)                 262,656   
dense_1 (Dense)                      (None, 10)                  5,130     
==================================================================
Total params: 14,982,474
Trainable params: 14,982,474
Non-trainable params: 0
```

## Results
Baseline Model Performance
The baseline model was trained for 100 epochs, and it achieved a test accuracy of 75.23% and a test loss of 0.7434.
The VGG16 model significantly outperformed the baseline. It was trained for 15 epochs and achieved a test accuracy of 92.90% and a test loss of 0.2430.

The VGG16 model's results, as expected, were far superior as it had been pre-trained on classification tasks for 2-3 weeks and hence simply needed to be fine-tuned for our task. 

Accuracy and loss of the Baseline Model:

![image](https://github.com/user-attachments/assets/ef165663-1015-45cf-a246-c78acbf0b050)
![image](https://github.com/user-attachments/assets/c4368a90-26e3-4b2c-b289-f83fb3ea4367)

Accuracy and loss of the VGG16 model:

![image](https://github.com/user-attachments/assets/cfabaf1b-d55c-4087-b5f0-cacd5be3b100)

## Conclusion
This project demonstrates the power of transfer learning, especially when working with image datasets that are not excessively large. The baseline CNN model achieved a solid performance, but the VGG16 pre-trained model significantly improved accuracy, reducing the error by over 17%. This highlights the effectiveness of using pre-trained models for image classification tasks, particularly when computational resources are limited.

## Further Improvements
Running the VGG16 model for more extended training periods and epochs could significantly reduce the loss further, as it was only trained for 15 epochs.
Experimenting with other architectures since Transfer learning can access many other pre-trained models, which could be worth exploring.

Further Improvements for the base model:
Running for more extended periods can significantly reduce losses, as the trend for loss was still downwards as we kept increasing epochs.
Make the model more complex if you have the computing power by adding more Convolutional and Pooling layers. 

## License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
