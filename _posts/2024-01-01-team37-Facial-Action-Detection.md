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

### Experiments
We attempted to train a model using JAA-Net that is public on BP4D dataset. With limited resources, we had to scale down the dataset, but we were successful to train JAA-Net and obtained its performance metrics.
#### Dataset
We were able to obtain a subset of BP4D dataset.BP4D-Spontaneous dataset is a database of videos of facial expressions in a group of young adults. The participants in the video clips are around 18-29 years of age from various ethnicity backgrounds. The expressions they displayed in these videos are induced from natural stimulus. Facial features are tracked in both 2D and 3D domains. The entire dataset is about 2.6TB in size. We utilized the metadata associated with AU activation and facial landmark information included with the dataset. [2] The sub-dataset we use contains 125884 images sampled from frames of videos, and groud truth labels of the presence of 12 AU labels and facial landmarks. The dataset we obntained was missing AU activation labels and facial landmark information for some pictures, causing a mismatch between images and labels.  This issue was discovered during training when the model displayed low loss but low accuracy during testing with new images. Because of lack of knowledge to properly label these images, we determined to exclude them from the dataset. The final dataset used contained 83924 images for training and 45809 images for testing. Training was done locally on a laptop with a RTX 3070Ti as it requires a large amount of memory and time, and it was not realistic to do on a Google Cloud VM with the few credits that were available. Because of limited resources, training set was kept small compared to the test set. 
#### Methods And Results
Some minor modifications to the code base was required to work with our specific dataset, but nothing with JAA-Net's structure were changed. The model was trained on the training set for 12 epochs with learning rate originally set at 0.00007 for first  that  0.000096 and decays down to 0.000024. Training took more than 50 hours, which underscores a significant drawback of this model: it requires a lot of resources and time to train. This is an older model using CNN and attention mechanisms, and they can be difficult to train. 

The model was then tested on the test set, which yielded the following results:

| Epoch | F1 Score Mean | Accuracy Mean | Mean Error | Failure Rate |
|-------|:-------------:|:-------------:|:----------:|:------------:|
|   1   |    0.521639   |    0.709132   |  0.046760  |   0.003956   |
|   2   |    0.527294   |    0.775433   |  0.044694  |   0.002002   |
|   3   |    0.534872   |    0.783711   |  0.040380  |   0.000715   |
|   4   |    0.534648   |    0.787174   |  0.040828  |   0.000763   |
|   5   |    0.541575   |    0.761680   |  0.040338  |   0.000643   |
|   6   |    0.556645   |    0.776259   |  0.040349  |   0.000667   |
|   7   |    0.555009   |    0.770363   |  0.040505  |   0.000620   |
|   8   |    0.553778   |    0.764504   |  0.040265  |   0.000596   |
|   9   |    0.551945   |    0.760742   |  0.039970  |   0.000572   |
|  10   |    0.549276   |    0.764444   |  0.040630  |   0.000596   |
|  11   |    0.550392   |    0.764053   |  0.040060  |   0.000596   |
|  12   |    0.552339   |    0.767260   |  0.039748  |   0.000620   |

This

## Reference
[1] Li, W., Abtahi, F., Zhu, Z., & Yin, L. (2017, May). Eac-net: A region-based deep enhancing and cropping approach for facial action unit detection. In 2017 12th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2017) (pp. 103-110). IEEE.

[2] Yin, L., Wei, X., Sun, Y., Wang, J., & Rosato, M. J. (2006, April). A 3D Facial Expression Database For Facial Behavior Research. In 7th International Conference on Automatic Face and Gesture Recognition (pp. 211-216)1. IEEE.


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
