---
layout: post
comments: true
title: NeRFs for Synthesizing Novel Views of 3D Scenes
author: Colin Melendez
date: 2024-03-22
---


> In the domain of generative images, NeRFs are a powerful tool for generating novel views of 3d scenes with an extremely high degree of detail. here, we will review the basics of Neural Radiance Field (NeRF) models, look at how they can be used to generate novel views, and investigate an optimization to the original design with the KiloNeRF model to see how we can improve on some of it's shortcomings.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Background - NeRFs

In recent years, Neural Radiance Fields (NeRF) have emerged as a groundbreaking approach in computer vision and graphics for creating 3D scene reconstructions from a set of 2D images. The original paper - “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis” by Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, et al [1] described a novel method by which a fully connected neural network could be trained to represent a space, and then queried to return the color of pixels in the space when rays are projected from an arbitrary camera location. 

<>

**The main innovation of the KiloNeRF model is this:**
 Instead of representing the entire scene with a single large MLP, the scene is instead represented by many smaller MLPs. The way in which this is done is that the scene is divided into a coarse grid of voxels, and each voxel in that grid has a small MLP assigned to represent that section of space. The image below provides a god visualization of this:

![NeRF-vs-KiloNeRF]({{ '/assets/images/team48/nerf-vs-kilonerf-architecture.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 6. NeRF vs KiloNeRF conceptual comparison* [2].


## Building our own KiloNeRF

My completed implementaion and training data is on my github as a jupyter notebook: [https://github.com/ColinMelendez/KiloNeRF-Implementation](https://github.com/ColinMelendez/KiloNeRF-Implementation)


## References

[1] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934). *2020. accessed March 2024*

[2] Christian Reiser, Songyou Peng, Yiyi Liao, Andreas Geiger. ["KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs"](https://arxiv.org/abs/2103.13744). 2021. *accessed march 2024*

Code for KiloNeRF model derived from:

[3] https://github.com/bmild/nerf

[4] https://github.com/creiser/kilonerf

[5] https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code

---
