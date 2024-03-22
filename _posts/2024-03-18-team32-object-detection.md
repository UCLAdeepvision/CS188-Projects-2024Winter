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

## Introduction
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Approach 1

## Intermediate Geometric Representation based Method

Another popular method of regressing 3d pose from 2d images is through the extraction of intermediate geometric representation. The best performing model of this class is the Ego-Net from 2021 [2]. 


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


### Result

Overall, this model out performs most other models (including Deep3DBox) by a solid 3-5% across many different categories. 

![EGO10]({{ '/assets/images/32/ego10.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

## Method

## Results

## Discussion

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[2] Shichao Li and Zengqiang Yan and Hongyang Li and Kwang-Ting Cheng,  et al. "Exploring intermediate representation for monocular vehicle pose estimation" *CVPR*. 2021.

---
