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


## OpenPifPaf

### What is OpenPifPaf?
OpenPifPaf is an open-source library for human pose estimation, focused on high performance on a low computational budget. This is often used for real-time predictions in autonomous vehicles such as cars and delivery robots. In the context of vehicles, these predictions quickly give the vehicle a more accurate representation of where pedestrians are, so the vehicle can plan its pathing further in advance. OpenPifPaf models use a bottom-up approach to allow for the extremely fast predictions needed for high-speed vehicles.

![OpenPifPaf example]({{ '/assets/images/team46/openpifpaf-example.jpeg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 3. An example pose estimation image from an OpenPifPaf model* [3].

### Backbone Architecture
OpenPifPaf's architecture is designed to emphasize speed while maintaining comparable accuracy to other HPE implementations. This is done by downsampling agressively in the backbone and CAF stages of the encoder, and by using lightweight backbones to begin with. The feature maps from the encoder are then fed into the decoder to output the final predictions.

![OpenPifPaf Architecture]({{ '/assets/images/team46/openpifpaf-architecture.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 4. Visualization of the OpenPifPaf model* [4].

### Loss Design
After the final prediction(s) are output by the decoder, loss is calculated on the prediction(s). Similar to MMPose, OpenPifPaf includes multiple parts in its loss calculation to account for different features of the predictions. Classification loss is binary cross entropy loss with a focal loss modification term, w. Focal loss modification increases the magnitude of loss on predictions that the model is least confident about, focusing learning on the harder parts of the dataset. For loss on the location and scale of each joint, Laplace loss is used:

![OpenPifPaf Loss]({{ '/assets/images/team46/openpifpaf-loss.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 5. OpenPifPaf loss equations* [4].

### MMPose Advantages
- Efficiency: OpenPifPaf is designed to operate on a relatively low computational budget, since it will be run at real-time on many frames per second. This also allows OpenPifPaf to run on lower-quality hardware.
- Accuracy on Crowded Images: Partially attributed to its loss design, OpenPifPaf excels in handling occlusions and crowded scenes, maintaining accuracy even with partial visibility of subjects. Also, OpenPifPaf uses a bottom-up approach, making execution on crowded images more efficient
- Simplicity: Compared to other frameworks such as MMPose, OpenPifPaf is easier to implement due to its shorter list of features. 

### MMPose Disadvantages
- Versatility: OpenPifPaf is specialized for its specific use cases, and will not outperform other models in many other cases.
- Performace: Since OpenPifPaf emphasizes speed, it often falls behind when compared to other models' accuracy. 

## Conclusion
Overall, human pose estimation is a very important field in computer vision, with many useful applications in the present and future. The two open-source approaches discussed in this post offer good solutions to HPE, and are still being developed to this day.


## References
Please make sure to cite properly in your work, for example:

[1] Nanonets. "A 2019 guide to Human Pose Estimation with Deep Learning." *https://nanonets.com/blog/human-pose-estimation-2d-guide/*. 2019.
[2] MMPose. *https://github.com/open-mmlab/mmpose*.
[3] OpenPifPaf. *https://github.com/openpifpaf/openpifpaf*.
[4] Kreiss, Bertoni, Alahi, et al. "OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association" *ArXiv. /abs/2103.02440*. 2021.

---
