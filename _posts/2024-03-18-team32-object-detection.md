---
layout: post
comments: true
title: 3D Bounding Box Estimation Using Deep Learning and Geometry
author: UCLAdeepvision
date: 2024-03-18
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

# Introduction
3D object detection is a challenging task in computer vision where the goal is to identify and locate objects in 3D environments based on their shape, location and orientation. 

Application fields include autonomous vehicles, robotics, and augmented reality.

In our project, we explore methods to recover 3D bounding box and orientation from monocular, 2D RGB images. We specifically focus on a classic paper in the field, "3D Bounding Box Estimation Using Deep Learning and Geometry". 



# Data Set

Most datasets consists of 2D images/videos that contain additional information such as Distance, Elevation, and Azimuth of the camera which is relevant for evaluating 3D Bounding box generated from 2D images/videos. 

The additional information is often recorded in the form of point-clouds measured using LIDAR technology.

Alternatively, there are tools which can be used to map CAD models of objects onto 2d images. 

![DATA1]({{ '/assets/images/32/data1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

## KITTI

One of the most popular benchmark dataset in 3D Object Detection is the KITTI dataset from 2012. 

Kitti stands for Karlsruhe Institute of Technology and Toyota Technological Institute. 
The dataset includes: Various sensor modalities, High resolution RGB, 3D LIDAR data, GPS Localization data (tying objects to GPS coordinates). 

For 3D Object detection, frames are selected to maximize the amount of cluttering objects
There have been various work done by other researchers to label the data further. 

The label processing is quite difficult, as labelling needs to be added on a pixel-wise and point-wise basis.

![DATA2]({{ '/assets/images/32/data2.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
<!-- ### Metric for Evaluation: Bounding Box

The official 3D Metric various across different dataset varies
AP used, where a 2D bounding box is compared against the 3D Bounding box

The Precision x Recall is a curve for each class where precision is Y axis and Recall is X axis.

We want Precision and recall both to be high or hug the top right corner. The area under the curve is used to summarize performance.

AUC is often used to calculate Average Precision, a very common metric used to measure other segmentation tasks. mAP is the mean AP across different object classes

<!-- IMAGE -->
<!-- 
### Metric for Evaluation: Orientation

KITTI uses “Average Orientation Similarity”, which is calculated using the cosine similarity between predicted orientation and GT orientation.

There is also metrics that measures model performance at detecting center of a 3D box bounded object

IMAGE --> -->




# Models

## XYZ Based Model

## Intermediate Geometric Representation Based Method

Another popular method of regressing 3d pose from 2d images is through the extraction of intermediate geometric representation. The best performing model of this class is the Ego-Net from 2021 [2]. This class of model is also similar to Deep3DBox, as they both try to regress 3d pose from 2d images.


### Past works
Many previous models attempt to directly map 2d pixels to angular representation of 3d poses. The drawbacks of these previous models is that local appearance alone in a 2d image is not sufficient to determine vehicle pose. 

![EGO1]({{ '/assets/images/32/ego1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
<!-- ![EGO2]({{ '/assets/images/32/ego2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"} -->

Furthermore, extracting 3d pose directly from 2d pixels is a highly non-linear and difficult problem. 

![EGO4]({{ '/assets/images/32/ego3.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### IGR

Inspired by the representational framework of vision introduced by Marr [3], Ego-Net explicitly defines and coerces the model to first learn some intermediate geometric representations. 

![EGO6]({{ '/assets/images/32/ego5.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

#### IGR Regression target

The model attempts to regress the orientation relative to the center of a cuboid. The regressed, correctly oriented cuboid is then projected back to the image plane. The 2D coordinate of the projection is the desired IGR.

![EGO6]({{ '/assets/images/32/ego6.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### Custom Loss Function

The lost function used for this model consists of heatmap loss, 2d and 3d. 

![EGO7]({{ '/assets/images/32/ego7.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

Another novel loss used in Ego-Net is cross ratio loss function. 

![EGO8]({{ '/assets/images/32/ego8.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

Since cross ratio is a projection invariant, the IGR's cross-ratio should be the same as the predicted 3d bound boxes' cross ratio. 

The use of the cross ratio loss function allows for self-supervised learning using only the cross ratio loss without a 2d ground truth. 

![EGO9]({{ '/assets/images/32/ego9.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


## 3D Bounding Box Estimation based Method

"3D Bounding Box Estimation Using Deep Learning and Geometry, Mousavian et al. 2017" starts with a 2D bounding box, and estimates 3D dimensions/orientation using that.

The model utilizes a Hybrid discrete-continuous loss function
Geometric constraints from 2D bounding box
for training. 

![EGO10]({{ '/assets/images/32/3d1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


The model's primary goal is to attempt to regress 3D Object Dimensions from 2d bounding boxes. As a part of the loss function, 

### 3D Dimension Regression

One of the first goals of the model is to regress 3d object dimensions from the 2d bounding boxs. 

The model makes the assumption that the 2d bounding boxes provided by the model is as tight as possible, and each side of the 2d box touches the projection of 1 or more of the 2d box's corners. 

With these assumptions in mind, the model attempt to derive a set of mathematical constraints through coordinate projection. 

![EGO10]({{ '/assets/images/32/3d2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


To do the actual regression for bounding box, the model uses a network of CNNs. After successfully regressing the bounding box, the model attempts to find the center point T of the bounding box that minimizes re-projection error. 

![EGO10]({{ '/assets/images/32/3d3.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

Some simplifying assumptions made about the model is that it is upright, and that there is no roll or pitch. 

### Multibin CNN Module

3D Orientation must be understood in the context of local orientation $\theta_l$ and camera angle $\theta_c$. 

In the images below, the global orientation doesn't change, even though the camera view changed. We thus try to regress $\theta_l$. ??

![EGO10]({{ '/assets/images/32/3d4.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


To take advantage of existing models, the Multibin CNN module is added onto a pre-trained VGG network. Orientation angle is discretized into several possible bins. Estimate confidence that true angle belongs in each bin (column 3),
necessary correction to θray (column 2), and dimensions based on the ITTI object category (column 1). Bin with max. confidence is selected during inference

![EGO10]({{ '/assets/images/32/3d5.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


### Multi bin Loss Function

Orientation Loss:

![EGO10]({{ '/assets/images/32/3d7.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


Localization Loss:

![EGO10]({{ '/assets/images/32/3d8.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


Dimension Loss:

![EGO10]({{ '/assets/images/32/3d9.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


Total Loss:

![EGO10]({{ '/assets/images/32/3d10.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


### Results

The model was able to beat stereo/semantic segmentation models. It was able to:

- 1st place for Average Orientation Estimation, Average Precision on KITTI Easy/Moderate datasets

- 1st place on KITTI Hard in Orientation Score (excludes 2D bounding box estimation)

- Beats SubCNN in other metrics (e.g. 3D box IoU)
3-8 deg of orientation error across KITTI Easy/Moderate/Hard

![EGO10]({{ '/assets/images/32/3d11.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


The model was also able to demonstrate the "Attention" where it focuses on features such as tires and mirrors to determine orientation. 

![EGO10]({{ '/assets/images/32/3d12.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

# Methods

# Results


# Discussion

Test

## Ego-net vs Deep3dBox

Ego-net (the IGR based method), being based on the work of the earlier 3D Bound box estimation based method, performed better.

![EGO10]({{ '/assets/images/32/ego10.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

Ego-net was able to out perform the 3D bounding box estimation based method because it took advantage of IGR which was a development found from other pose-estimation fields in CV. Inspired by developments from human pose estimation, the model was re-architecture to encourage the learning of IGR via custom loss functions. This reduced the difficulty of mapping directly from 2d pixels to 3d orientation. 

Furthermore, the use of cross ratio loss function enabled the model to have some levels of self-supervised learning. In fact, results from the paper show that the model is able to learn pose from unlabelled data through cross-ratio loss alone, which is something Deep3dBox does not have. 

All in all, it make sense that Ego-net was able to out perform Deep3dBox, thanks IGR and the cross-ratio loss function. 


# Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[2] Shichao Li and Zengqiang Yan and Hongyang Li and Kwang-Ting Cheng,  et al. "Exploring intermediate representation for monocular vehicle pose estimation" *CVPR*. 2021.

---
