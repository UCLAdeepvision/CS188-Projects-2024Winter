---
layout: post
comments: true
title: Human Pose Estimation
author: Kyan Kornfeld
date: 2024-03-21
---


> Human pose estimation (HPE), a pivotal task in computer vision, seeks to deduce the configuration of human body parts from images or sequences of video frames. In this post, I will examine two approaches to human pose estimation: MMPose, the most common library for HPE, and OpenPifPaf, a more lightweight, efficient pose estimation model.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
In human pose estimation, a pose refers to an arrangement of human joints in a specific manner. HPE models represent these poses in images and, more common as of late, videos. Common applications of human pose estimation include action recognition, motion capture, movement analysis, augmented reality, sports and fitness, and robotics. 

![Human pose estimation example]({{ '/assets/images/team46/figure1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. An example of human pose estimation* [1].

## Top-down vs Bottom-up
Top-down HPE models run an object detector to place each person in a bounding box, then estimate the joints within the bounding box. Bottom-up human pose estimation estimates the joints first, then group them together into a pose. Top-down approaches tend to be more accurate than bottom-up ones, but bottom-up models are more scalable because runtime increases proportionally to the number of people in the image in top-down HPE. This post will discuss one bottom-up approach, OpenPifPaf, and one approach that can be top-down or bottom-up, MMPose.

## MMPose

### What is MMPose?
MMPose is an open-source toolbox developed by the Multimedia Laboratory of The Chinese University of Hong Kong, that offers comprehensive, flexible, and extensible pose estimation solutions. MMPose is one part of a very large computer vision library, OpenMMLab, that includes models for object detection, semantic segmentation, generation, and many other tasks.

![MMPose]({{ '/assets/images/team46/mmpose-logo.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### Backbone Architecture
MMpose offers a very wide range of neural networks as a backbone for pose estimation. The backbone is used for feature extraction: it is essentially a decoder. Over 20 backbones are offered, including AlexNet, MobileNet, ResNet, ResNeXt, ShuffleNet, and more. These backbones vary in depth and thus, computations, so each network may have applications in which it is the most optimal.

![MMPose Model Overview]({{ '/assets/images/team46/mmpose-model-overview.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. Visualization of the MMPose model* [2].

### Loss Design
After the input data passes through the backbone, the model makes prediction(s) on the pose(s), and loss is calculated on the prediction(s). MMPose supports many algorithms for calculating loss, and the ability to use multiple loss functions to include loss components for different parts of HPE. These loss components will then be weighted averaged together to calculate the total loss. MMPose allows your implementation to calculate loss as a combination of associative embedding, bounding box, classification, heatmap, logit distance, and regression losses.

These losses are combined in a loss wrapper class. Below is an example of a loss wrapper containing heatmap and associative embedding loss:

```
heatmap_loss_cfg = dict(type='KeypointMSELoss')
ae_loss_cfg = dict(type='AssociativeEmbeddingLoss')
loss_module = CombinedLoss(
    losses=dict(
        heatmap_loss=heatmap_loss_cfg,
        ae_loss=ae_loss_cfg))
loss_hm = loss_module.heatmap_loss(pred_heatmap, gt_heatmap)
loss_ae = loss_module.ae_loss(pred_tags, keypoint_indices)
```

### MMPose Advantages
- Versatility: MMPose supports a wide range of pose estimation tasks and algorithms.
- Community and Support: Part of the large and widely accepted OpenMMLab ecosystem, using MMPose ensures active community support and continuous development.
- Extensibility: Easy integration with other OpenMMLab projects for tasks like object detection and semantic segmentation.

### MMPose Disadvantages
- Complexity: The wide range of features can overwhelm new users. The documentation on MMPose is very long, as MMPose supports so many features.
- Resource Intensive: MMPose algorithms require substantial computational resources to train and test, potentially limiting accessibility for individuals or organizations with limited hardware.





## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
