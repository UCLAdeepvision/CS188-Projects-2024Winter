---
layout: post
comments: true
title: Facial-Action-Detection
author: UCLAdeepvision
date: 2024-01-01
---


<!-- > This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know. -->


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## EAC-Net (CNN approach to Facial Action Unit detection)
### Structure
The EAC-Net is a convolutional neural network (CNN) designed for Facial Action Unit (AU) detection, integrating enhancing and cropping features to focus on specific facial expressions. It is composed of three main components: a fine-tuned pre-trained VGG 19-layer network, enhancing layers (E-Net), and cropping layers (C-Net). This structure allows for detailed feature extraction and learning, tailored to the nuances of facial expressions and AU detection.

![EAC-Net-Architecture]({{ 'assets/images/team37/EAC-Net-Architecture.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Architecture of EAC-Net* [1].

#### Finetuning Network
The base of the EAC-Net utilizes a fine-tuned pre-trained VGG 19-layer network. The lower-level convolutional layers (groups 1 and 2) are retained with their original parameters for extracting basic visual features. In contrast, the parameters of the higher-level convolutional layers (groups 3 and 4) are fine-tuned specifically for AU detection. This approach ensures the network has a solid foundation in understanding the input images at both basic and complex levels.
#### E-Net
The enhancing layers, or E-Net, are added atop the high-level convolutional layers of the VGG network. These layers employ an attention map based on facial landmark features to enhance the learning process, focusing specifically on areas of interest related to AUs. The goal of the E-Net is to extract features with more valuable information for AU detection, drawing a parallel to the structure of Residual Net but with a focus on generating enhanced features.
#### C-Net
The cropping layers, known as C-Net, focus on precise facial regions by cropping sub-features from ten selected interest areas of the feature map. These areas then undergo further processing with upscale layers and convolutional layers to deepen the learning on each facial region. C-Net ensures the network pays attention only to relevant regions for AU detection, enabling deeper contextual understanding.
### Loss Function
The loss function for EAC-Net is designed to handle the multi-label binary classification problem inherent in AU detection, where multiple AUs may be present simultaneously. It employs cross-entropy to measure loss, adjusted with constants to prevent excessively large values and stabilize training.

$$
\text{Loss} = -\Sigma \left( l \cdot \log\left( \frac{p}{1.05} + 0.5 \right) + (1 - l) \cdot \log\left( \frac{1.05 - p}{1.05} \right) \right)
$$

$$l$$ represents the label (1 for the presence of an AU and 0 for its absence), and $$p$$ denotes the predicted probability for the AU's presence. The constants 1.05 and 0.5 are used to adjust the loss function to prevent the loss values from becoming excessively large, thus stabilizing the training process.

### Attention Map
The attention map is a critical component of the E-Net, designed to give more attention to individual AU areas of interest. It is generated based on the distance to the AU center, employing the Manhattan distance formula to calculate the weight of each pixel. This approach ensures that the enhancing layers focus more precisely on the areas of the face most relevant to AU detection, improving the accuracy and effectiveness of the EAC-Net.

![EAC-Net-Attention-Map]({{ 'assets/images/team37/EAC-Net-Attention-Map.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Attention Map of EAC-Net* [1].

Formula for calculating weight of each pixel:
$$
w = 1 - 0.095 \cdot d_m
$$

$$d_m$$ is the Manhattan distance to the AU center

## Reference
[1] Li, W., Abtahi, F., Zhu, Z., & Yin, L. (2017, May). Eac-net: A region-based deep enhancing and cropping approach for facial action unit detection. In 2017 12th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2017) (pp. 103-110). IEEE.



<!-- ## Main Content
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
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

--- -->
