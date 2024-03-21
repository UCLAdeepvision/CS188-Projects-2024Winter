---
layout: post
comments: true
title: Facial-Action-Detection
author: UCLAdeepvision
date: 2024-01-01
---


<!-- > This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know. -->
## Introduction
### Background Introduction: Application value of facial action detection in natural human-computer interaction, emotional analysis, and other fields.
Facial Action Detection is an important aspect in improving human and technology interactions. The ability to recognize and analyze facial expressions is essential for sentiment analysis and other fields that require understanding a user's emotions. ⁤⁤Facial action detection technology allows for more interactions by recognizing and responding to facial emotions.This effectively reduces the communication gap between individuals and machines.
![Illustration of Facial Pose Detection and Transformation]('../assets/images/team37/Facial-Pose-Detection-and-Transformation.png')
Visualizing the process helps when we talk about the usefulness of facial activity recognition in applications. A practical example of the change that facial action detection technology provides is shown in Figure 1. The technique starts with a source image and uses unique pose displacements that are applied to generate a target image with a changed posture. This procedure illustrates how capable these systems are in detecting, predicting, and modifying facial expressions, bringing up new possibilities for complex interactions between humans and machines. In applications that include sentiment analysis to current time responsive models in virtual worlds, the accuracy and flexibility shown here are essential.

### Research Objective: Compare the effectiveness and possible applications of various facial motion detection methods with the aim of comparing their performance and application potential.
The main objective of this project is to evaluate and contrast different facial motion detection techniques in order to get further insight into their potential. We will dive into further detail about the obstacles of every approach in this project. Important factors including processing speed, accuracy of detection, and potential losses will also be evaluated. The technique may be applied to many different fields, mainly aim to aid in sentiment analysis. For example, Healthcare, the automated analysis of human emotions allows for early and accurate diagnosis of depression, anxiety, and other mental illnesses, or new forms of entertainment (like games, movies) that will adapt to your emotions. Others like educational technologies that can detect students’ emotional responses and personalize their learning experiences, improving engagement. Together, these applications show how AU Detection is not just a technical feat but a bridge to more empathetic and responsive technology. With this research, we are hoping for insight into the importance of facial action detection techniques, as well as their advantages and disadvantages, and provide a deeper understanding of the significance that each approach performs in certain applications.

### Project Structure: Summarize the contents of the report, including discussion and comparative analysis of the three approaches.
In this project, we will be focused on the implementation and evaluation of three different facial motion detection methods: basic CNN model, Joint Attention Aware Network, and Method Based on ResNet50. First, we will cover the discussion of different approaches. The Basic CNN Model is the first model to be analyzed, along with its technical details, application cases, advantages and limitations. The Joint Attention Aware Network (JAA-Net) is then being analyzed in detail, similar to the process for CNN mode, emphasizing on its usefulness and layout in facial action detections and the utilization of attention procedures. After that, we will look at the ResNet50-Based Models, which focus on its architecture and how to enhance model performance by introducing graph attention mechanisms. Next, we will be looking at the comparison analysis, and how well each approach performs in terms of accuracy, speed in real time, and losses. Lastly, we will talk about the practical application of facial action detection. Our experimental design, result analysis, problems and possible solutions will also be discussed.

## Overview of Facial Action Detection Techniques
### Basic Principles: Brief introduction of the key concepts of facial action units and detection.
Facial Action Units (FAUs) are the basic concepts used to analyze and interpret a variety of emotions and facial movements. ⁤⁤It also serves as the basis for comprehending human face expressions. ⁤⁤The expression of emotions depends on the movement of a particular group of facial muscles, which corresponds to each FAU.
![Illustration of Facial Pose Detection and Transformation]('../assets/images/team37/Facial-Action-Detection')
A structured flowchart can help us see and comprehend the complex process of facial action recognition as we dive deeper into the details (see Figure 2). The flow chart illustrates how the source picture is transformed, with a focus on action unit (AU) detection and position adjustments, to create a target image that includes the detected face motions. Complex algorithms that analyze facial features and recognize patterns and changes related to various expressions are required to detect these action units. ⁤⁤In fields like emotion identification, for example, this technique is essential because precisely localizing these units might improve our comprehension of human interactions and emotions. ⁤⁤As technology develops, techniques for identifying these action units get more complex, utilizing sophisticated computer models to boost the precision and dependability of real-time analysis. 

### Technological Development: Overview of the evolution from early algorithms to current deep learning methods.
When we look at facial action detection technology from the very beginning until now, we can see that this technology has evolved significantly over time, moving from simple algorithms to complex deep learning techniques. At first, Paul Ekman and Wallace V. Friesen developed FACS in the 1970s even before computers, which only looked at facial muscle actions and descriptions. Then with the rise of computer vision, researchers began to explore its application in AU detection. First, the technique advanced with the development of machine learning methods such as Support Vector Machines (SVM). Then, ⁤the learning algorithms experienced a revolution with the development of CNNs. Its deep layers allowed for the direct learning of complicated facial expressions from large amounts of data. ⁤Then, JAA-Net provides a new approach that improves the model's response to complicated emotions by integrating attention mechanisms. It focuses on relevant facial areas. Finally, ResNet50 improves the accuracy of facial motion detection by integrating Deep Residual Learning, which enables deeper network training without decreasing from gradient descent. All these techniques help the evolution of facial detection from a very simple algorithm to now with more complex, deep learning methods.

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
