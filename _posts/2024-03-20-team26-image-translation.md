---
layout: post
comments: true
title: Post Template
author: UCLAdeepvision
date: 2024-03-22
---

> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.

<!--more-->

{: class="table-of-content"}

- TOC
  {:toc}

## Main Content

Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Introduction

Our team is investigating style transfer in the context of image-to-image translation.

## Background

### Image-to-image translation

The goal of image-to-image translation is learning a mapping between input image and the output image. The dataset used for this problem is typically two sets of images we want to learn the mapping between. These datasets come in the form of

1. Paired: the dataset is tuples of image in set 1 and corresponding image in set 2
2. Unpaired: the dataset just has two sets of images without 1-to-1 correspondence.

## Cycle-GAN

Generative adversarial network (GAN) are deep learning frameworks that relies on a generator G and a discriminator D. Cycle GAN introduces **cycle consistency** (similar to language translation, where a sentence in English when translated to German then translated back should be the same as English). 

To preserve cycle consistency, we want to make sure when our network translates an image, we can translate it back to get a similar image to the original image. In order to do this, we train two GANs together, Gan 1 $(G, D_g)$ translating from style 1 to style 2. Gan 2 $(F, D_f)$ translating from style 2 to style 1. We additionally introduce a normalization term on the input image $I$ and the $F(G(I))$, the input image translated twice.


## Basic Syntax

### Image

Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
_Fig 1. YOLO: An object detection method in computer vision_ [1].

Please cite the image if it is taken from other people's work.

### Table

Here is an example for creating tables, including alignment syntax.

|      | column 1 | column 2 |
| :--- | :------: | -------: |
| row1 |   Text   |     Text |
| row2 |   Text   |     Text |

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

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016.

---
