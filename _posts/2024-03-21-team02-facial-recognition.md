
---
layout: post
comments: true
title: Deep Neural Networks for Facial Recognition
author: Aidan Wittenberg, Delia Ivascu, Rafi Rajoyan
date: 2024-03-21
---

# Table of contents
1. [Introduction](#introduction)
2. [Classical Challenges](#classicalchallenges)
    1. [Challenge 1: name](#challenge1)
3. [Deep Learning to Address Challenges](#deeplearningaddresschallenges)
	1.  [Deep Dense Face Detector (DDFD)](#ddfd)
	2. [FaceNet](#facenet)
4. [References](#reference)

# Introduction <a id="introduction"></a>

This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.

## Classical Challenges <a id="classicalchallenges"></a>
The first paragraph text

### Challenge 1: name<a id="challenge1"></a>
This is a sub paragraph, formatted in heading 3 style

## Deep Learning to Address Challenges <a id="deeplearningaddresschallenges"></a>
The second paragraph text

## Solutions <a id="deeplearningaddresschallenges"></a>

### Deep Dense Face Detector (DDFD) <a id="ddfd"></a>
ddfd text

### FaceNet <a id="facenet"></a>
facenet text

## References <a id="reference"></a>
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.


---
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