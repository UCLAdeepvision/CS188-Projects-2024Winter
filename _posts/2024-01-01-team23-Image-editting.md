---
layout: post
comments: true
title: Text Guided Image Editing using Diffusion
author: Jack He, Allen Wang, Yuheng Ding, James Jin
date: 2024-03-07
---


> In this study, we explore the advancements in image generation, particularly focusing on DIFFEDIT, an innovative approach leveraging text-conditioned diffusion models for semantic image editing. Semantic image editing aims to modify images in response to textual prompts, enabling precise and context-aware alterations. We conduct a thorough comparison between DIFFEDIT and various traditional and deep learning-based methodologies, highlighting its consistency and effectiveness in semantic editing tasks. Additionally, we introduce an interactive framework that integrates DIFFEDIT with BLIP and other text-to-image models to create a comprehensive end-to-end generation and editing pipeline. Moreover, we delve into a novel technique for text-guided mask generation within DIFFEDIT, proposing a method for object segmentation based solely on textual queries.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
### What is Image Editing
Image editing refers to the process of altering or manipulating digital images. It encompasses a wide range of tasks, including color correction, cropping, resizing, and retouching. It has a variety of applications, such as photography, graphic design, and digital art.

The recent surge in image editing technology, driven by advancements in deep learning, has revolutionized the field. Deep learning-based methods have showcased impressive abilities in generating and modifying images, paving the way for a diverse array of applications. These include image inpainting as evidenced by (Yu et al., 2018), style transfer as explored by (Jing et al., 2019), and semantic image editing, a concept furthered by (Couairon et al., 2022). These developments have birthed a novel paradigm in image editing, particularly notable in text-guided image editing.

This innovative approach is able to find significant applications in the fashion industry. Designers now have the capability to input descriptions of desired styles, colors, and patterns, and watch as AI systems adapt existing designs to these specifications. This method offers a more efficient pathway for testing and developing new design concepts.

Furthermore, the impact of text-guided image editing can extend into augmented reality (AR) and virtual reality (VR) environments. Here, users can interact with virtual objects, either superimposed onto the real world or within entirely virtual settings. Text-guided image editing introduces an intuitive interface, allowing users to modify these virtual objects—altering aspects like color, size, texture, or position—through simple text or voice commands. This not only enhances user experience but also broadens the scope of interactivity within these digital environments.

### Semantic Image Editing
Semantic image editing, a subset within the broader realm of image editing, concentrates on refining and modifying images in accordance with textual descriptions. This method is effective for executing precise and complicated modifications, as it enables users to articulate their desired changes in natural language. The essence of semantic image editing lies in its complexity; it demands the model to not only comprehend the textual description but also to apply suitable alterations to the image accurately.

This challenge, however, is what endows semantic image editing with its vast potential for application across a wide spectrum of image editing tasks. Its capabilities extend to other image editing tasks such as the removal of objects, colorization of images, and the transfer of styles. This versatility highlights the significance of semantic image editing in the comtempory landscape of digital image manipulation, which is why we have chosen to focus on this area in our study.

## Previous Works
### Classical Approches

### Diffusion Approches

## Dive into DiffEdit

## Experiment and Benchmarks

## Our Ideas
### End-to-End Generation and Editing
### Text Guided Diffusion Based Object Segmentation

## Ethics Impact
## Conclusion
## Code
## Reference


<!-- ### Image
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

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016. -->

---
