---
layout: post
comments: true
title: Hand Gesture Recognition
author: Aidan Jan, Jacob Ryan, Randall Scharpf, Howard Zhu
date: 2024-03-21
---

> This project explores the subject of hand gesture recognition, a subset of computer vision and human pose estimation that aims to learn and classify hand gestures.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Overview
### What is Hand Gesture Recognition
Hand gesture recognition aims to identify hand gestures in the context of space and time. It is a subset of computer vision whose goal is to learn and classify hand gestures.

From the dictionary:

```
Gesture (noun): a movement of part of the body, especially a hand or the head, to express an idea or meaning.
```

Hand gesture recognition or gesture recognition in general has many applications. It can be used for communcation such as ASL translation, interfacing with electronic devices, and even computer animation. Our research here outlines hand gesture recognition in general but the focus of our implementation will be interfacing with electronic devices.

### Objectives

To begin understanding hand gestures we must have data to look at. Data sources often come from cameras, 3D camera systems, and motion tracking setups. This gives us visual information about what is in the scene and the context around it.

The goal of hand gesture recognition is to take the input data, detect the presence of a hand, and then extract the meaning (or lack there of) behind the movement. Outputs from a model like this could be classifications of gesture types, bounding boxes, skeleton wire frames, or just text outputs.



## Current Methods of Object Detection

### Early Models of Hand Gesture Recognition
Currently, the most common methods of human pose detection do not use computer vision.  Rather, it uses a method referred to as Passive Optical Motion Capture, which involves people (or animals) wearing suits with reflective markers on them that cameras can easily track.

[INSERT IMAGE, mocap]

This is convenient for animations, game development, and movies since it provides accurate 3-dimensional points that can be easily analyzed later.  However, many other use cases are appearing where reflective markers are not practical, such as in sports analysis, virtual reality development, wildlife research, or sign language translations, just to name a few.  Pose Detection with deep learning offers a non-invasive method for real-time body tracking.

Hand gesture tracking is really a subset of human pose detection, which is a subset of object or feature detection.  In fact, each hand gesture may be treated as their own separate 'object' for the computer to recognize.  As a result, hand trackers tend to use the same technology as human pose detectors and general object detectors - just with a few extra specializations.

[INSERT IMAGE, different hand gestures]

The earliest object detectors use a structure called an R-CNN, or a region-based convolutional neural network.  Although these work for processing recorded videos, they take too long to run to be practical for real-time video processing.  This lead to developments on the neural network and the advent of the Fast R-CNN and Faster R-CNN which are approximately 9.6 times and 210 times faster than R-CNN, respectively.  More recently, another neural network model named YOLO emerged, which runs faster than Faster R-CNN at the cost of accuracy.

### Architecture of YOLO
YOLO, an acronym for “You Only Look Once” is a single-pass neural network (e.g., each image is processed only once) used for object detection.  

Figure 1
https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e

It accomplishes this task by essentially merging the region proposal network and the general object classification neural network, making it faster than Faster R-CNN.

YOLO first passes the input image through a CNN for feature extraction, followed by multiple fully connected layers which predict class probabilities and bounding boxes.  The image is then divided into a grid of squares, where each square, or grid cell, is associated with a set of class probabilities and bounding boxes (See Figure 1).  Bad bounding box selections are then filtered out using a process referred to as non-maximum suppression (discussed later), then finally predicted bounding boxes with their class predictions are returned.

The image processing pipeline YOLO uses is massively efficient compared to its predecessors Faster R-CNN and Fast R-CNN.  However, its main drawback is that it is not as accurate, especially with small objects like hands.  Because of the step where the image is divided into a grid, the smallest object that can be classified by YOLO is the size of one grid cell.  As a result, although YOLO is great for classifying larger objects very quickly, it is not particularly efficient or accurate for hand gesture recognition, since the number of grid cells must be increased.

## Google Media Pipe

### SSD with Retina-Net Influence
Single Shot Detectors (SSDs) is a single pass algorithm for generating bounding boxes, much like YOLO. However, at the price of a couple extra convolutions per image is a great increase in accuracy, especially with varying scales for how big the bounding box should be. 

Like YOLO, it splits the image into regions, and has a certain set of default sized bounding boxes (known as anchors) that it applies to each region. But, unlike YOLO, instead of having plenty of differently sized anchors, SSDs have comparatively few - this is because SSDs don’t just have one convolution step, but multiple, and thus can make predictions for bounding boxes at each “scale”. This allows SSDs to make more accurate predictions for objects that have varied sizes, and much better at predicting large objects (due to the large amount of information after multiple convolutions). 

But, predicting hands is a difficult task. There’s a reason that AI and artists alike are famously bad at generating/drawing hands, and that’s because there’s so many different poses and possibilities for hands to be in. So, the Mediapipe architecture made a couple changes to the standard SSD to adjust it to be able to detect hands with over 95% accuracy! (Source)

One of the large changes they made was to add a feature-pyramid network(source)as well as implementing Focal Loss(paper), inspired by the work of Retina-Net. 

### Non-Maximum Suppression

### Regression Model


## Future Applications / Research?


## Conclusion






























#
