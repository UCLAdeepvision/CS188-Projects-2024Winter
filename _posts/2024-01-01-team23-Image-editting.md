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

#### Editing
#### Inpaiting
#### OutPainting
#### Style Transfer

### Image Editing Application

## Previous Works
### Classical Approches
### Diffusion Approches

## Dive into DiffEdit
In many cases, semantic image edits can be restricted to only a part of the image, leaving other parts unchanged. However, the input text query does not explicitly identify this region, and a naive method could allow for edits all over the image, risking to modify the input in areas where it is not needed. To circumvent this, DIFFEDIT propose a method to leverage a text-conditioned diffusion model to infer a mask of the region that needs to be edited. Starting from a DDIM encoding of the input image, DIFFEDIT uses the inferred mask to guide the denoising process, minimizing edits outside the region of interest.

Before delving into the intricate workings of DiffEdit, it's essential to lay the groundwork by understanding some foundational concepts that it builds upon: DDIM and Classifier-Free Guidance. These elements are vital for grasping how DiffEdit achieves its targeted editing prowess.

### Denoising Diffusion Implicit Models (DDIM)
DDIM, short for Denoising Diffusion Implicit Models, is a variant of the diffusion models used for generating or editing images. One problem with the DDPM process is the speed of generating an image after training. DDIMs accelerate this process by optimizing the number of steps needed to denoise an image, making it more efficient while maintaining the quality of the generated images (with a little quality tradeoff). It does so by redefining the diffusion process as a non-Markovian process. The best part about DDIMs is they can be applied after training a model, so DDPM models can easily be converted into a DDIM without retraining a new model. 

![DDIM]({{ '/assets/images/28/DDPMvsDDIM.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig # Graphical models for diffusion (left) and non-Markocivian (right) inference models.* [#].

The reverse diffusion process in Denoising Diffusion Implicit Models (DDIM) serves as a fundamental mechanism allowing these models to reconstruct or generate images by methodically reversing the diffusion process that gradually transforms an image into random noise. This process is central to understanding how DDIM and, by extension, technologies like DiffEdit function, enabling them to create detailed and precise image edits or generate images from textual descriptions. Here's a simplified overview of the reverse diffusion process, avoiding deep mathematical complexities (for which [this post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling) provides an excellent derivation).

![DDIM]({{ '/assets/images/28/DDIM_Denoising_Formula.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig # DDIM_Denoising_Formula.* [#].

Importantly, instead of randomly walking back through the noise levels, DDIM uses a deterministic approach to carefully control the denoising path. Thus we no longer have to use a Markov Chain since Markov Chains are used for probabilistic processes. We can use a Non-Markovian process, which allows us to skip steps.

### Classifier-Free Guidance
Classifier-Free Guidance is a technique used to steer the generation process of a model towards a specific outcome without relying on a separate classifier model. In traditional guided diffusion models, a classifier is often used in tandem with the diffusion model to ensure the generated images adhere to certain criteria. However, Classifier-Free Guidance simplifies this by eliminating the need for a separate classifier. Instead, it modifies the diffusion process itself to guide the generation towards the desired output. This is achieved in two key steps:

#### Training Phase: 
![CFG]({{ '/assets/images/28/With_without_class.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig # Noise estimation model with and without class (null).* [#].

During training, the model learns to generate outputs based on a wide range of inputs, including those without specific class labels. With a probability p_uncond, we make some of the classes null classes. This approach enables the model to understand the underlying distribution of the data more broadly, rather than being constrained to specific labeled classes.

![CFG]({{ '/assets/images/28/cfg_training.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig # Training with classifier-free guidance.* [#].

#### Generation Phase: 
![CFG]({{ '/assets/images/28/cfg_sampling_noise.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig # Noise model parameterization for classifier-free guidance.* [#].

The noise prediction requires two forward passes of the same image, zₜ. One forward pass calculates the predicted noise not conditioned on a desired class, and the other calculates the predicted noise conditioned on the desired class information. When generating new content, CFG employs a technique called "guidance scale" or "temperature," which adjusts the strength of the model's predictions towards certain attributes or themes. By tweaking this scale, users can control how closely the output adheres to the desired attributes without the need for an external classifier.

![CFG]({{ '/assets/images/28/cfg_sampling.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig # Sampling with classifier-free guidance.* [#].

By utilizing DDIM for efficient and precise encoding of the input image and incorporating Classifier-Free Guidance to direct the editing process without the need for additional classifiers, DiffEdit sets the stage for sophisticated image editing. These technologies allow DiffEdit to infer a mask of the region to be edited based on the text input, ensuring that changes are made only where intended. This approach not only preserves the integrity of the unedited portions of the image but also provides a high level of control and specificity in the editing process.

### Three Steps of DiffEdit
With this knowledge ready, let's take a look at the three steps of DiffEdit.

#### Step one: Mask Generation
#### Step two: Encoding
#### Step three: Decoding with Mask Guidance

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
