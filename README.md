# Machine Learning Project: Dog Breed Identification
A machine learning project that identifies a dog's breed given its image. 
This notebook builds an end-to-end multi-class image classifier using TensorFlow and Google Colab.
This is the final project developed by Daniel Bourke in his 3 month course:
* [Complete A.I. Machine Learning and Data Science: Zero to Mastery](https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/)

## Tools used: Python (TensorFlow, Keras, Pandas, Numpy, Matplotlib), Google Colab

# Goals: 
Learn TensorFlow, deep learning and transfer learning, apply computer vision model from research paper.

## 1. Problem
Use computer vision to classify dog photos into different breeds.

## 2. Data
The data is from Images of dogs from Stanford Dogs Dataset (120 dog breeds, 20,000+ images).

http://vision.stanford.edu/aditya86/ImageNetDogs/

## 3. Evaluation
Multi Class Log Loss between the predicted probability and the observed target. For each image in the test set, predict a probability for each of the different breeds.

## 4. Features
Some information about the data:
- There are 120 breeds of dogs (so there are 120 different classes).
- There are around 20,000+ images in the data set 

## 5. Model

Use the [`tf.keras.applications.efficientnet_v2.EfficientNetV2B0()`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2B0) model from the 2021 machine learning paper [*EfficientNetV2: Smaller Models and Faster Training*](https://arxiv.org/abs/2104.00298) from Google Research and apply it to our own problem.

This model has been trained on [ImageNet1k](https://en.wikipedia.org/wiki/ImageNet) (1M+ images across 1000 different diverse classes, there is a version called ImageNet22k with 14M+ images across 22,000 categories). It has a good baseline understanding of patterns in images across a wide domain.

We'll see if we can adjust those patterns slightly to our dog images.  What this means is we'll use most of the model's existing layers to extract features and patterns out of our images and then customize the final few layers to our own problem. This kind of transfer learning is often called **feature extraction**.  It is a technique where you use an existing models pre-trained weights to extract features (or patterns) from your own custom data. You can then used those extracted features and further tailor them to your own use case.
