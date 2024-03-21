---
layout: post
comments: true
title: 3D Scene Representation from Sparse 2D Images
author: Krystof Latka, Patrick Dai, Srikar Nimmagadda, Andy Lewis
date: 2024-03-10
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction to 3D Reconstruction

## Problems with classical approaches
### Ambiguity
Baseline SfM only does not account for scaling. 
As seen below, if any linear transformation Q is applied to the projection matrix, 
its inverse can be applied to the points matrix, implying that transformed represetations
may be judged as equivalent.


$$ x = {PX} = {(PQ^-1)(QX)}$$

        x - Correspondence
        P - Projection matrix
        X - Points matrix
        Q - Transformation



<br>
According to the different types of 3d representations seen below:

![Different ambiguities](../../../assets/images/Team29/ambiguity.png)

We find that we can only generate projective reconstructions of an input scene.
We would need additional information for affine/similar/euclidean representations.


### Occlusion and Limited Viewpoints
Given a sparse set of images: <br>
- For nearby cameras: may get large triangulation errors for far-away points <br>
- For far-away cameras: correspondence can be missed <br>

SfM depends purely on input images to construct a geometric representation of an input scene, so given limited vewpoints 
it may struggle to capture comprehensive 3D structures. It will often miss details or occluded areas in the scene, as shown below:

![Missed structures due to limited viewpoints](../../../assets/images/Team29/occlusion.png)

The highlighted structures are occluded by larger structures in the central cathedral at certain camera points, leading to missed correspondence and failed point cloud reconstruction. 
This can be mitigated with better camera point planning (i.e. if camera positions were planned to include regions between the central cathedral and sides stuctures, occlusion could be limited),
but this is difficult to acheive given limited viewpoints.

### Textureless Surfaces
Low-texture surfaces also pose a unique challenge for SfM based approaches. SfM hinges on identifying corresponding points between different camera positions,
and textured objects make the challenge geometrically easier. Low-texture surfaces, on the other hand, have features that are harder to identify, which can add difficulty to 
the correspondence process. Below is an example where low-texture, repetitive surfaces lead to mismatches.

![low texture](../../../assets/images/Team29/low texture.png)


Non-lambertian surfaces are a unique low-texture case which can cause correspondence mismatching. Reflective or transparent surfaces display different
features depending on the observer's position, so such surfaces can lead to either missed correspondence (where a surface cannot be matched with its previous images)
or incorrect correspondence (which is often more disastrous).

<img src="../../../assets/images/Team29/reflection.png" alt="low texture" width="500"/>
Above is an image depicting how a single reflective surface has an inverse surface which could be incorrectly matched depending on the camera position.


### Noise Defects
Global SfM struggles with outliers and noisy input which can cause errors in feature point matching <br>
Outliers: <br>
- Incorrect correspondence can be detrimental during feature tracking, where features in one frame can be mapped incorrectly to features in another
- During bundle adjustment (where the collection of a point's positions in different frames are mapped into a 3D mesh), this can cause extreme errors with features  <br> <br>

General noise between frames can also result in epipolar line misestimations. Epipolar lines are used to map camera positions relative to each other, where one camera
will estimate the degree and location of a ray between another camera's optical center and image point. Failures to calculate epipolar lines can create warped estimations of a scene.

Pictured below: Epipolar line misestimation with noise <br>
<img src="../../../assets/images/Team29/noise.png" alt="low texture" width="400"/> <br>

    Green - ground truth
    Red - added gaussian noise
    Blue - noise + colinearity
    Yellow - noise + coplanar knowledge


## Deep Learning Comes to Rescue
### Instant NGP
### Zero-1-to-3
### 3D Gaussian Splatting

## Conclusion

## Code Repositories

## References
Please make sure to cite properly in your work, for example:

[1] Liu, G., Klette, R., Rosenhahn, B. (2006). Collinearity and Coplanarity Constraints for Structure from Motion. In: Chang, LW., Lie, WN. (eds) Advances in Image and Video Technology. PSIVT 2006. Lecture Notes in Computer Science, vol 4319. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11949534_2

[2] Alsadik, Bashar. (2014). Guided close range photogrammetry for 3D modeling of cultural heritage sites. 10.3990/1.9789036537933. 

[3] S. Lazebnik. (2019). Strcuture From Motion. http://www.cs.cmu.edu/~16385/

---
