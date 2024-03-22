---
layout: post
comments: true
title: Computer Vision for Brain Tumor Detection
author: Vikram Nagapudi, Justin Downing, Raj Jain
date: 2024-3-20
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
In this paper, we explore methods of detecting malignant and benign brain tumors using modern deep learning and computer vision architectures. A brain tumor occurs when abnormal cells form around the brain, and can be classified as either a malignant or benign growth. A malignant tumor - which, as it sounds, is very fatal - grows much faster than its counterpart, and may spread into a person's nervous system. On the contrary, benign tumors grow much slower, and are very unlikely to spread to other areas. As a result, being able to classify whether a tumor is of each category is of life-or-death importance for a brain cancer patient. As machine learning architectures are evolving, there have been numerous approaches to automating this classification, to aid surgeons and doctors in cancer treatment, and to ultimately save more lives.

## Classical Approach
Brain tumors are detected through imaging - through black and white MRI scans - and physical detection (the use of excisions to examine the skull and brain). MRI scans are non-invasive and critical, as they serve as the main interpreter for cancer doctors in their classification task. Today, cancer patients must wait for long periods of time to get a result back from their doctor regarding their health. These doctors also deal with another issue, other than the number of patients to look after, which is the data available regarding existing brain tumors. The combination of both these problems leads us to consider machine classification of brain tumors as our solution to patient wait times and use of historical data. However, as we are about to explore, the classification and segmentation of tumors is no easy task.

## Modern Approaches to Classification
The reason for machine automation is twofold: doctors wish to first classify the tumor as malignant or benign, and segment the tumor itself given an MRI image. The first implementations of this classification faced a common issue: the lack of annotated training MRI images. In beginning approaches, researchers used single 2D black and white MRI scans, meaning image preprocessing and usage required only standardization in one channel (as opposed to RGB channels). More modern classification techniques have been trained on MRI images that emulate 3D scans - more on this later.

### CNN-Based Brain Tumor Detection
The first approach we will introduce involves the use of CNNs for 2D MRI tumor detection. The researchers (Hosseini, et. al) proposed this pipeline:
1. Perform Data Augmentation to Increase Training Samples

![DATAAUG]({{ '/assets/images/teamrajjustinvik/mriimgs.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Augmenting existing MRI images to create extra training data* [1].

The researchers mention that the original dataset used for training was only 200 images, in comparison to the thousands of images used by classic image classification architectures. To address this issue, the researchers performed rotations and changed the angle of the image, as can be seen in the figure above. The original 200 or so images became almost 1400 using this technique - for every 1 image, the researchers were able to generate 7 new images from image augmentation.

2. Preprocess Images Before Inputting into DL Model

Given that the images given to the researchers were greyscale 2D MRI scans, the preprocessing of the images before input into the CNN had three steps. First, the researchers cropped each image to get rid of unwanted black portions surround the brain. They then applied Gaussian blurring to each image to reduce the random noise, using an inbuilt function in cv2. As mentioned, they did this as the Gaussian generates a weighted average of each pixdl's area, standardized by the standard deviation of the central pixel in the Gaussian kernel. They then performed a binary conversion to differentiate the brain (white in the scan), with the black background.

3. The CNN

The CNN architecture used by the researchers was quite elementary. They performed 2D convolution with ReLU as the activation function, then fed this output to the dense layer, which used a sigmoid function as the activation function - to perform binary classification (malignant of benign). Though the CNN was simple, the results achieved by this model were much better than its predecessors.


Model Performance: 


### Grey Wolf Optimization

In more modern architectures, as we will see through this classification model developed by Eldin, et. al, researchers are able to leverage "3D" MRI scans to classify the existence of a brain tumor. While our last approach used a single-layer greyscale MRI image, most radiology images are given to doctors in a "3D" format - involving stacks of 2D MRI scans that resemble a higher dimensional image. This approach builds on papers like the one above in terms of increasing model accuracy during the training stage itself, which we will explore below.

The model methodology is as follows: 
![GWOVER]({{ '/assets/images/teamrajjustinvik/greywolfoverview.png' | relative_url }})

We see that the model differs from our first exploration due to its focus on hyperparameter optimization - thus lending the name "Grey Wolf."

#### What is Grey Wolf Optimization?

The researchers of this approach mention that previous brain tumor classification models are able to return very high classification accuracy scores, but with a couple problems: lack of segmentation, consideration of brain position, and low accuracy. For this classification problem, a "low accuracy" is any accuracy below 99%. This is due to both the lack of training data, and the importance of classifying a malignant tumor. To address these issues, the researchers focused on hyperparameter optimization as such:

First, the hyperparemeters the researchers focused on include optimization momentum, drop rate factor, learning rate, epochs, learning rate schedule, and the L2 regularization factor - the basic hyperparameters for a CNN model.

Now for the optimization itself, the use of ADSCFGWO - or adaptive dynamic sin-cosine fitness grey wolf optimization - allows for the selection of best training hyperparameters by emulating the hierarchy of a wolf pack. 

![GWDIAGRAM]({{ '/assets/images/teamrajjustinvik/greywolfdiagram.png' | relative_url }})


They start by initializing each "wolf" as a potential solution to the optimization problem - or a set of specific values for their hyperparameters. They then evaluate each "wolf" on the model, creating their objective or "fitness" function - the accuracy on the validation set of the data. Based on this fitness function, they take the top three wolves, and update the hyperparameter position to the direction of the "prey" - the most optimized solution. Thus, the wolves that achieve higher model accuracy "lead" the other wolves - as their hyperparameters are updated slower than the other possible hyperparameter value sets, which have to be updated more steeply to follow their "alpha wolves". Once this grey wolf model converges, the researchers then perform their CNN-based classification - more on that below. 

#### Segmentation of Brain Tumor Images

![UNET]({{ '/assets/images/teamrajjustinvik/unetarch.png' | relative_url }})
*Fig 3. Overview of U-Net Segmentation Architecture* [3]

The researchers also tackled the lack of segmentation architecture for brain tumor imaging in their approach. By using U-Net, which is medical imaging's most used software for imaging, to perform 3D segmentation, they were able to achieve this goal. We are able to visualize the architecture above.

![UNET]({{ '/assets/images/teamrajjustinvik/segresults.png' | relative_url }})
*Fig 4. Results of 3D U-Net Segmentation* [4].

Previous 2D U-Net architectures used a similar encoder-decoder approach we learned during class. The encoder performed downsampling on the given image, using convolution layers and max pooling with stride 2, as given by the image above. The decoder or "expanding pathway" performed upsampling using transposed convolution, along with more convolutional layers and a 1x1 layer at the end to extract 64 features from the original input image for segmentation. The 3D architecture used by the researchers builds on this by using 3D convolution and max-pooling, and using batch normalization for a faster convergence and compute time.

#### Newer Brain Tumor Dataset

![GWOVER]({{ '/assets/images/teamrajjustinvik/trainimgs.png' | relative_url }})
*Fig 5. Example of training images in the BraTS Dataset* [5].

The last consideration of this paper is the availability of newer segmented imaging data. In 2021, the BraTS challenge was introduced, and a dataset of around 1200 images with annotated labels and hand-drawn segmentation data from neurologists was provided to the public. As opposed to the paper above, which started with 200 images and augmented the training dataset to include over 1200, this approach was given a much larger dataset to help prevent overfitting of data, a common issue with brain tumor detection.

#### Results of Grey Wolf Model
JUSTIN WORK HERE

### Segmentation Using 3D CNN Architecture

#### Considerations of a 3D Network

The reseachers of our new approach, Casamitjana, et. al, propose a 3D CNN model, as opposed to the common 2D models introduced above. But in order to train a 3D model, the number of parameters and memory usage consideration is very important. The researchers focused on optimizing the depth of their network along with sampling to emulate the 2D computational performance while increasing the accuracy for 3D MRI images.

#### Issues with Pooling Techniques

The researchers mention that pooling techniques (such as the one used in the U-Net architecture above) do help with compute time. However, they reduce the ability of the segmentation model to capture fine-grained details. They tackle this issue by using dense inference on smaller image patches to obtain the finer features, while using convolutional layers and pooling to obtain general features for segmentation. This approach is called hybrid training. 

#### Hybrid Training CNN (Model 1)

![HYBRiD]({{ '/assets/images/teamrajjustinvik/hybridcnn.png' | relative_url }})
*Fig 6. Hybrid Training Methodology for 3D CNN* [6].

As mentioned before, the researchers needed to find a way to reduce computational complexity while capturing both fine-grained and coarse features. As can be seen, there are two prediction blocks for each stage of the CNN. The prediction blocks toward the right of the model are for lower-resolution images after more pooling and convolutional layers. These blocks are used for coarse details and identifying the position of the tumor. The prediction block toward the left is used for fine-grained features, as the input image is much higher in resolution, so the CNN is able to capture texture and edges at this stage.

#### 3D U-Net Convolution (Model 2)

![unet3d]({{ '/assets/images/teamrajjustinvik/3dunet.png' | relative_url }})
*Fig 7. Approach 2: 3D U-Net for Segmentation* [7].

In the section on Grey Wolf optimization, we were introduced to 2D and 3D U-Net architectures for segmentation. This newer model proposed by our researchers builds on the U-Net architecture, again with the goal of capturing fine-grained and coarse features. They achieve this by concatenating the feature maps in each layer of the downsampling with their respective upsampling counterpart, as can be visualized by the "u concat v" in the diagram above. This concatenation allows the network to use contextual and local information to generate each layer output.   

#### Training Data and Evaluation Results

The researchers in this approach used the BraTS 2016 dataset, which was much more limited than the dataset we were introduced to in the above section. This older dataset contains about 200 annotated images for each case - malignant and benign - as opposed to the 1200 images from before. As a result, the researchers did have to deal with overfitting issues, but still achieved impressive results:

JUSTIN RESULTS HERE

## Conclusions and Future Work



## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
