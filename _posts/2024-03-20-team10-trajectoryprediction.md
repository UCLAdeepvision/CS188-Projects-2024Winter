---
layout: post
comments: true
title: Trajectory Prediction
author: Yu-Chen Lung, Edward Ng, Warrick He, Alan Yu
date: 2024-03-22
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Alan inserts his introduction to the problem here. Can base off of past ones.

## Papers and Approaches

# Social LSTMs

[Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)

Humans have the ability to “read” one another, predicting each other’s general motions, each obeying a set of untold rules that help us predict the motion of other humans. However, trying to predict this motion with machines has proven to be a difficult task. Previous methods have either used functions to predict behavior or only consider other people in close proximity.

> Social LSTMs tackle the problem of trajectory prediction, more specifically pedestrian trajectories, and allow for anticipating more distant interactions.

#### Social LSTM Architecture

Social LSTMs base themselves on the traditional LSTM architecture, which has proven to be very useful for sequence prediction tasks such as caption generation, translation, video translation, and more. In their specific model, the authors of Social LSTM introduce a few key ideas. Firstly, each trajectory is represented by its own LSTM model. Secondly, the LSTMs are linked to each other via a special layer called a social pooling layer.

![SocialLSTM]({{ '/assets/images/team10/sociallstms.png' | relative_url }})
{: style="width: 400px; max-width: 100%; text-align: center;"}

# VectorNet

# Other

## Comparisons
<!-- 
## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

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
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/). -->

## Reference
Please make sure to cite properly in your work, for example:

[1] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei and S. Savarese, "Social LSTM: Human Trajectory Prediction in Crowded Spaces," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 961-971, doi: 10.1109/CVPR.2016.110.

---
