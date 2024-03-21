
---
layout: post
comments: true
title: Deep Neural Networks for Facial Recognition
author: Aidan Wittenberg, Delia Ivascu, Rafi Rajoyan
date: 2024-03-21
---

> Facial recognition is the technology of identifying human beings by analyzing their faces from pictures, video footage or in real time. Facial recognition has been an issue for computer vision until recently. The introduction of deep learning techniques which are able to grasp big data faces and analyze rich and complex images of faces has made this easier, enabling new technology to be efficient and later become even better than human vision in facial recognition.


# Table of contents
1. [Introduction](#introduction)
2. [Classical Challenges](#classicalchallenges)
    1. [Challenge 1: name](#challenge1)
3. [Deep Learning to Address Challenges](#deeplearningaddresschallenges)
4. [Solutions](#solutions)
	1.  [Deep Dense Face Detector (DDFD)](#ddfd)
	2. [FaceNet](#facenet)
5. [References](#reference)



# Introduction <a id="introduction"></a>

Facial recognition is the technology of identifying human beings by analyzing their faces from pictures, video footage or in real time. Facial recognition has been an issue for computer vision until recently. The introduction of deep learning techniques which are able to grasp big data faces and analyze rich and complex images of faces has made this easier, enabling new technology to be efficient and later become even better than human vision in facial recognition.

### Why do we care about facial recognition?

Facial recognition has gained significant attention and relevance in today's digital landscape due to its multifaceted applications and implications across various domains. From enhancing security measures to streamlining user authentication processes, facial recognition provides unparalleled convenience and efficiency. We can see facial recognition integrated in law enforcement, retail, healthcare, and even social media platforms. This highlights its versatility and potential to revolutionize how we interact with technology and each other. 


Why is facial recognition a difficult task?

Facial recognition proves to be a complex task due to the variety of factors involved in real-world scenarios. These include the variability of face angles, expressions, lighting conditions, and even aging effects. A common challenge arises from the clarity of images. When images lack clarity, facial features become less discernible, leading to increased pixelation. As a result, deep learning algorithms require higher resolution input images to accurately identify and analyze facial characteristics. 

In order to address these challenges, we need to develop advanced algorithms and techinques tailored to the nuances of facial recognition tasks.

### Multi-view face detection
Current state-of-the-art approaches for this task require annotation of facial landmarks or annotation of face poses. They also require training dozens of models to fully capture faces in all orientations, angles, light levels, hairstyles, hats, glasses, facial hair, makeup, ages.

### Face recognition processing flow

![YOLO]({{ 'assets/images/2/DDFD/Process-flow.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. replace this* [1].

![YOLO]({{ 'assets/images/2/DDFD/processflow2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. replace this* [1].

### Face recognition tasks

These are the most important tasks in face recognition:


1. **Face Matching**: find the best match for a given face

2. **Face Similarity**: find faces that are most similar to a given face

3. **Face Transformation**: generate new faces that are similar to a given face

4. **Face Verification**: a one-to-one mapping of a given face against a known identity (e.g. is this the person?)

5. **Face Identification**: a one-to-many mapping for a given face against a database of known faces (e.g. who is this person?)

## Classical Challenges <a id="classicalchallenges"></a>
The first paragraph text

### Challenge 1: name<a id="challenge1"></a>
This is a sub paragraph, formatted in heading 3 style

## Deep Learning to Address Challenges <a id="deeplearningaddresschallenges"></a>
Deep learning and neural networks provide a number of benefits over traditional means of facial detection and recognition addressing the common challenges of the classical approaches. Neural networks learn from a large amount of data and are therefore more robust, efficient, performant, and simple. 
AI is generally divided into three main branches: artificial intelligence, machine learning, and deep learning. Historically, up to 2010, applied AI was synonymous with machine learning (ML), which involves creating models that learn from historical data to predict future outcomes. A notable limitation of ML is the need for specialized logic to transform raw input—like images—into a set of handpicked features, such as color histograms that numerically represent color distributions, enabling the ML models to learn and make predictions based on these numeric features [3].


![Benefits of DNNs]({{ 'assets/images/2/Benefits_of_DNNs/ML_vs_DL.png' | relative_url }})
*Fig 1. Benefits of Deep Learning over Machine Learning in Feature Extraction* [2].
{: style="width: 400px; max-width: 100%; margin: auto;"}

The advent of DNNs marks a shift to the third significant phase of practical AI—deep learning. This transition is marked by the use of vast datasets, greater computational power, and a crucial advancement: machines learning autonomously. Deep neural networks (DNNs), comprising interconnected nodes akin to neural pathways in the human brain, excel in identifying and decoding complex patterns. They do this by layering nodes: initial layers decode simple elements, such as an object's outline, while the more advanced layers discern intricate attributes like texture. Such sophisticated pattern recognition opens the door to a variety of innovative uses across industries.

One specific form of DNNs called convolutional neural networks (CNNs) are specially designed to process and classify objects in images. Convolutional neural networks consist of many convolution and pooling layers. At a high level, a convolution layer is responsible for detecting specific features in an image. A convolution layer accomplishes this task by applying a set of learnable filters (called kernels) across an image. Each kernel is designed to activate strongly when it detects a specific feature at a certain spatial position in the image. As the kernel moves across the image, it produces what is called a feature map which represents the presence and intensity of that feature in different regions of the input image. Repeatedly applying convolution layers allows a network to capture highly specific features in a hierarchical fashion making CNNs exceptional at handling visual information.

![Convolution Layer]({{ 'assets/images/2/Benefits_of_DNNs/conv_layer.png' | relative_url }})
*Fig 2. The primary calculations executed at each step of convolutional layer* [2].
{: style="width: 400px; max-width: 100%; margin: auto;"}

The second key layer in a convolutional neural network is the pooling layer. Pooling layers are interspersed between convolutional layers to reduce the size of the representation thus reducing the number of parameters and computational load the network creates. 

![Pooling Layer]({{ 'assets/images/2/Benefits_of_DNNs/pool_layer.png' | relative_url }})
*Fig 3. Three types of pooling operations* [2].
{: style="width: 400px; max-width: 100%; margin: auto;"}

The last and final type of layer that is necessary to understand for a CNN is a “fully connected” (FC) or “linear” layer. After the initial convolution and pooling layers have detected features the FC layers are used to interpret those features and perform classification of the subject of the image. The last layer in the network is commonly a FC layer which has as many nodes as classes in the task (for example 5 neurons for a 5-class classification task). The neurons in the FC layer have full connections to all activations in the previous layer, as their inputs are computed as a weighted sum and a bias offset. Finally a softmax activation function is commonly applied which simply converts the output of the FC layer into normalized probabilities for each class. The class with the highest probability is often taken as the model’s prediction. 

![FC Layer]({{ 'assets/images/2/Benefits_of_DNNs/fc_layer.png' | relative_url }})
*Fig 3. Fully connected layer* [2].
{: style="width: 400px; max-width: 100%; margin: auto;"}

CNNs have a wide set of benefits over traditional methods and have transformed facial detection and recognition via deep learning methodologies like Deep Dense Face Detector (DDFD) and FaceNet. These advanced models offer unparalleled efficiency, robustness, and adaptability, as outlined below.

**Multi-view Face Detection:** Deep learning models significantly streamline the facial detection process. Unlike traditional approaches that rely heavily on facial landmarks and poses annotations, these models autonomously learn to recognize faces from many orientations. This autonomy in learning enables the detection of faces at various angles and positions, thus simplifying the training process and enhancing the model's utility across a diverse range of scenarios.

**Handling Occlusions and Variations:** The hierarchical structure of neural networks enables the detection of partially covered faces and those with significant variations. Early layers in the network might identify basic outlines, while deeper layers discern more complex features such as textures or expressions. This hierarchical learning approach ensures that even when parts of the face are obscured or altered, the network can still identify the presence of a face with incredible accuracy.

**Efficiency and Simplification:** One considerable advancement brought by deep learning in the realm of facial detection is the conversion of fully connected layers into convolutional ones. This transformation allows models to process images of any dimensionality, effectively reducing the network's complexity. The result is a more streamlined, efficient process that accelerates the facial detection without compromising accuracy, making deep learning models superior in speed and performance compared to their predecessors. Moreover, the automated learning of customized features makes the creation of neural networks much more simple.

**Robustness to Changes:** The adaptability of neural networks to new, unseen images underscores their robustness. These networks generalize learnings from training data to accurately recognize faces under conditions not previously encountered, such as different environmental settings or lighting variations. This characteristic is particularly vital for applications requiring high levels of precision across various operational contexts.

**Improving with Data Augmentation:** Data augmentation techniques play a critical role in enhancing the diversity of training data, which in turn bolsters the model's robustness. By introducing variations in face size, orientation, and occlusion through methods such as cropping, flipping, and scaling, models are better prepared to generalize from training to real-world applications. This improvement is critical in maintaining high accuracy levels in facial detection across a wide array of situations.

**Continuous Improvement:** The field of deep learning is characterized by its capacity for ongoing refinement and adaptation. As new data becomes available, models can be retrained or fine-tuned, ensuring they stay at the forefront of facial detection technology. This ability to evolve makes deep learning models particularly valuable, offering solutions that remain effective as the landscape of challenges and requirements shifts.

## Solutions <a id="solutions"></a>

### Deep Dense Face Detector (DDFD) <a id="ddfd"></a>

[Deep Dense Face Detector](https://arxiv.org/pdf/1502.02766.pdf), short for "DDFD", is a method for multi-view face detection that doesn't rely on pose or landmark annotations.

Existing approaches require training multiple models or additional components like segmentation or bounding-box regression. DDFD, on the other hand, proposes a better, more efficient solution, using a single deep convolutional neural network (CNN). The proposed method achieves comparable or better performance to state-of-the-art methods without the need for complex annotations.

### FaceNet <a id="facenet"></a>
facenet text

## Final words
When thinking about facial recognition, it is important to keep in mind that alongside its great promise comes a variety of ethical, privacy, and societal concerns, prompting critical discussions on the ethical deployment and regulation of this powerful technology. As facial recognition continues to evolve and be integrated into different aspects of our lives, understanding its capabilities, limitations, and ethical considerations is crucial for shaping its responsible and equitable use in society.

## References <a id="reference"></a>

[1] Farfade, Sachin Sudhakar; Saberian, Mohammad; Li, Li-Jia. "Multi-view Face Detection Using Deep Convolutional Neural Networks." 2015.

[2] Alzubaidi, L.; Zhang, J.; Humaidi, A.J. et al. "Review of deep learning: concepts, CNN architectures, challenges, applications, future directions." J Big Data. 8, 53 (2021).

[3] Bommasani, Rishi et al. "Center for Research on Foundation Models (CRFM) at the Stanford Institute for Human-Centered Artificial Intelligence (HAI)." arXiv:2108.07258 [cs.LG]. (2022). https://doi.org/10.48550/arXiv.2108.07258.


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

---