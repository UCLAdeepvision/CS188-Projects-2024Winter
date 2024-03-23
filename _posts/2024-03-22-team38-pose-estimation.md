---
layout: post
comments: true
title: Pose Estimation
author: Josh McDermott, Ryan Carmack, Michael Reed
date: 2024-03-22
---


> The problem of Human Pose Estimation is widely applicable in computer vision—almost any task involving human interaction could benefit from pose estimation. Human Pose Estimation—occasionally shortened to just Pose Estimation—is the process of predicting and labeling the pose of a human body from a 2D image or video. In essence, the algorithm produces a model—the pose—of the person or people it observes.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
_DeepPose: Human Pose Estimation via Deep Neural Networks (DeepPose)_

DeepPose was one of the first models to attempt to be “holistic.” Prior to it, parts based models had been the main tool used to estimate poses. DeepPose considered the entire subject by passing the whole image through a 7 layer convolutional neural network and predicting a pose vector. It became the earliest model to embrace deeper learning for the problem of pose estimation, and generalized very well, outperforming the previous best models in almost every task. 

More specifically, this model architecture has the following sizes for its learnable layers, where C represents a convolutional layer, P represents a pooling layer, F is a fully connected layer, and LRN symbolizes a local response normalization layer. 

C (55 X 55 X 96) - LRN - P - C(27 X 27 X 256) - LRN - P - C(13 X 13 X 384) - C(13 X 13 X 384) - C(13 X 13 X 256) - P - F(4096) - F(4096)

Dropout regularization is performed in the F layers, at p=0.6 


As an aside, LRNs were layers that were introduced in AlexNet, and are modeled after a neural phenomena known as lateral inhibition (excited neurons “downregulate” neighboring neurons). They do this by normalizing around local parts of the input sequence. LRNs are very rare now.

In order to work within the fixed input size, the input images and the ground truth pose vectors are normalized with respect to a bounding box that is obtained by running a person detector on the images. This normalization is applied to both the images as well as the ground truth pose vectors. The images are cropped to a fixed size of 220x220.The cropped input images are regressed to a normalized pose vector, in
$$
\mathbb{R}^{2k}
$$
The loss then is calculated as the L2 loss between the ground truth normalized pose vector and this prediction. It thus aims to optimize:

$$
\arg\min_{\theta} \sum_{(x,y) \in D_N} \sum_{i=1}^{k} \left\lVert y_i - \psi_i(x; \theta) \right\rVert_2^2
$$

(Note that we can simply omit occluded joints from the sum.)

At the time, this above architecture yielded a then high number of parameters, about 40M, limiting their ability to increase the image size further. As such, it predicted poses on a relatively coarse scale. To attempt to give the model access to higher resolution portions of the image, they further regress joint locations in a series of cascading models described below.
After the first model, we are left with rough pose predictions based off of the initial bounding box. In each of the subsequent cascading stages, bounding boxes are deterministically formed around the joint predictions from the previous stage (yielding different bounding boxes per joint). Critically, the original image is then cropped around this bounding box, allowing a higher level of detail to be captured, and normalized joint locations are calculated similar to before. We then apply an identical architecture as used in the initial prediction in order to predict a displacement for each predicted joint from the previous stage, aiming to minimize this loss further with the more granular image. Additionally, they augment the data in these cascading stages, by combining the ground truth with a set of simulated ground truth predictions, formed via displacing the ground truth labels by an amount sampled from a normal distribution. This sampling distribution is formed to have the same mean and variance as the observed displacements across all training examples. For these cascading models then, DeepPose optimizes the following:

$$
\theta_s = \arg\min_{\theta} \sum_{(x,y_i) \in {D_A}^s} \left\lVert y_i - \psi_i(x; \theta) \right\rVert_2^2
$$

The training data is formed from LSP and FLIC, described in detail earlier in the report. It is further augmented by both randomly translated crops, as well as left/right flips.
The model was evaluated with different amounts of cascading through two different metrics. Firstly, PCP was used as it was the standard metric at the time, and the authors wanted the results to be comparable to others. Secondly, the authors chose to also evaluate what they call the Percentage of Detected Joints (PDJ), as it alleviated some of the concerns present with PCP and shorter limbs. PDJ is nearly identical to the metric referred to as PCK earlier in our report; it classifies as detected if the distance between the predicted and true joint is within a particular fraction of the torso diameter.

The results were quite impressive. In previous studies, certain joints tended to be classified better by certain models, likely due to the less holistic nature of earlier models. On the contrary, DeepPose was able to broadly outperform other models, regardless of the particular joint. This was seen in both PCP as well as PDJ evaluations. Furthermore, the results of cascading were significantly helpful. As shown in the table below, the initial DeepPose stage was still comparable to other leading models. But when the images were evaluated in a 3 stage model, they achieved results better than the prior state of the art. 

![deepposetable1]({{ '/assets/images/team38/deep_pose_table_1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. DeepPose: Results of DeepPose Cascading Stage Model vs Contemporaries* [1].

At the time, DeepPose was a strong achievement, however, it still had its challenges. The model is small by today’s standards, and had some common failure cases, such as flipping the left and right side of predictions when the image subject was photographed from behind. DeepPose clearly did not “solve” pose estimation, and evolution was still very necessary.


## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![deepposetable1]({{ '/assets/images/team38/deep_pose_table_1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. DeepPose: Results of DeepPose Cascading Stage Model vs Contemporaries* [1].

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
\mathbb{R}^{2k}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference

[1] A. Toshev and C. Szegedy, ‘DeepPose: Human Pose Estimation via Deep Neural Networks’, in 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014.

---
