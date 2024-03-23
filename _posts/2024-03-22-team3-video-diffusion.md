---
layout: post
comments: true
title: Text-to-Video Generation
author: Shan Jiang, Brandon Vuong, Seth Carlson, Joseph Yu
date: 2024-03-21
---

> In this paper, we will discuss diffusion-based video generation models. We will first do a preliminary exploration of diffusion, then extend it to video generation by examining Video Diffusion Models by Jonathon Ho, et al. We will then follow this with a refinement of video diffusion models by conducting a deep dive into Imagen Video, a high definition video generation model developed by researchers at Google. Through this paper, we aim to provide an overview of diffusion-based video generation, as well as rigorously cover a high definition refinement of the basic video diffusion model.

- [Introduction](#introduction)
- [Diffusion](#diffusion)
- [Video Diffusion Models](#video-diffusion-models)
  - [3D U-Net](#3d-u-net)
  - [Factorized Space-Time Attention](#factorized-space-time-attention)
  - [Video-Image Joint Training](#video-image-joint-training)
- [Imagen Video](#imagen-video)
  - [Cascaded Architecture](#cascaded-architecture)
  - [SR3: Mechanism of Super-Resolution Block](#sr3-mechanism-of-super-resolution-block)
  - [V-Prediction](#v-prediction)
- [Conclusion](#conclusion)

## Introduction

Text to video generation is a computer vision task that uses deep learning to create a video from a text description. It takes an input of a text script, like a story or description and ideally outputs a high definition video that encapsulates the content of the script. It uses natural language prompts to understand the content of the text for the video output. Here is an example using Sora, a video generation model developed by OpenAI:

<div style="text-align: center;">
  <video width="740px" controls>
    <source src="https://cdn.openai.com/sora/videos/italian-pup.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <p style="text-align: center;">
    Fig 1. Prompt: The camera directly faces colorful buildings in Burano Italy. An adorable dalmatian looks through a window on a building on the ground floor. Many people are walking and cycling along the canal streets in front of the buildings. [5]
  </p>
</div>

## Diffusion

### Forward Diffusion Process

In diffusion, the forward process entails the adding of noise to the datapoint. Diffusion models usually sample this noise from a Gaussian distribution according to a noise schedule, but other distributions are possible. Diffusion models usually have a Markov Chain structure (meaning that future events only depend on the current state). By the end of the process, the datapoint should resemble the Gaussian noise, while maintaining key features of the image, allowing for the backward process to detect and utilize those features to reconstruct the whole image.It is typical to use a cosine noise schedule which has this Markov structure: 

$$
q(\mathbf{z}_t|\mathbf{z}_s) = \mathcal{N}(\mathbf{z}_t; (\alpha_t/\alpha_s)\mathbf{z}_s, \sigma^2_{t|s}\mathbf{I})
$$


### Backward Diffusion Process

## Video Diffusion Models

### 3D U-Net

### Factorized Space-Time Attention

The denoising model is trained using a weighted mean squared error loss function where \( \hat{x}_\theta(z_t) \) represents is the estimate of the denoised data provided by the model with parameters.
$$
\mathbb{E}_{\epsilon, t} [ w(\lambda_t) \| \hat{x}_\theta (z_t) - x \|_2^2 ]
$$

### Video-Image Joint Training:

In this innovative training method, individual images are treated as single-frame videos. This is achieved by organizing standalone images into sequences that mimic the length of a video. The approach cleverly bypasses certain video-specific computations and attention mechanisms by employing masking techniques. This strategy enables the training of video models on extensive and diverse image-text datasets.

The outcome of joint training is the knowledge transfer from images to videos. Specifically, while training exclusively on natural video data allows the model to understand dynamics within natural environments, incorporating images into the training process enables it to learn about various image styles, including sketches, paintings, and more.

#### Examples of Video Diffusion

<div style="display: flex; justify-content: center;">
    <div style="flex: 33.33%; padding: 5px;">
        <img src="https://video-diffusion.github.io/assets/001.webp" alt="Image 1" style="width: 100%;">
    </div>
    <div style="flex: 33.33%; padding: 5px;">
        <img src="https://video-diffusion.github.io/assets/026.webp" alt="Image 2" style="width: 100%;">
    </div>
    <div style="flex: 33.33%; padding: 5px;">
        <img src="https://video-diffusion.github.io/assets/000.webp" alt="Image 3" style="width: 100%;">
    </div>
</div>

## Imagen Video

Imagen Video is a video-generation system based on a cascade of video diffusion models. It consists of 7 sub-models dedicated to text-conditional video generation, spatial super-resolution, and temporal super-resolution. Imagen video has the capacity to generate high definition videos (1280x768) at 24 frames per second for a total of 128 frames.
<!--
## Diffusion Models -->

## Loss Function

This model is trained using the below loss function
$$
\mathcal{L}(\mathbf{x}) = \mathbb{E}_{\mathbf{\epsilon} \sim \mathcal{N}(0,\mathbf{I}), t \sim \mathcal{U}(0,1)} \left[ \| \hat{\mathbf{\epsilon}}_{\theta}(\mathbf{z}_t, \lambda_t) - \mathbf{\epsilon} \|_2^2 \right]
$$

### Cascaded Architecture:

Cascaded Diffusion Models, introduced by Ho et al. in 2022, have emerged as a powerful technique for generating high-resolution outputs from diffusion models, achieving remarkable success in diverse applications such as class-conditional ImageNet generation and text-to-image creation. These models operate by initially generating an image or video at a low resolution and then progressively enhancing the resolution through a sequence of super-resolution diffusion models. This approach allows Cascaded Diffusion Models to tackle highly complex, high-dimensional problems while maintaining simplicity in each sub-model.

![Architecture]({{ '/assets/images/team3/architecture.png' | relative_url }})
{: style=" max-width: 100%;"}
_Figure 2: The cascaded sampling pipeline starting from a text prompt input to generating a 5.3-
second, 1280×768 video at 24fps. “SSR” and “TSR” denote spatial and temporal super-resolution
respectively, and videos are labeled as frames×width×height. In practice, the text embeddings are
injected into all models, not just the base model._ [2].

The figure above summarizes the entire cascading pipeline of Imagen Video:

- Text encoder : Encode text prompt to text_embedding
- Base video diffusion model
- SSR\*3 (spatial super-resolution): increase spatial resolution for all video frames
- TSR\*3 (temporal super-resolution): increase temporal resolution by filling in intermediate frames between video frames

### SR3: Mechanism of Super-Resolution Block:

The SSR (Super-Resolution via Repeated Refinement) and TSR blocks implement a mechanism for conditional image generation known as SR3. This method revolves around training on sets that comprise pairs of low-resolution (LR) and high-resolution (HR) images. The inputs for training include:

- A low-resolution image, denoted as x
- Noisy image y<sub>t</sub>, which is derived from High-Resolution image y<sub>0</sub> using the equation $$\bar{y}=\sqrt{\gamma}y_0+\sqrt{1-\gamma}\epsilon$$
- Noise variance $\gamma$, which correlates with the total number of training step T.

![Algorithm1]({{ '/assets/images/team3/algorithm1.png' | relative_url }})
{: style=" width:600px; max-width: 100%;"}
_Fig 3: Algorithm for diffusion._[3]

<p style="font-size: 14px">Remark: The meaning of the loss function here is to make the difference between the noise output by the model and the randomly sampled Gaussian noise as small as possible.</p>

Inference Process:

![Algorithm2]({{ '/assets/images/team3/algorithm2.png' | relative_url }})
{: style=" width: 600px ; max-width: 100%;"}
_Fig 4: Algorithm for iterative refinement._

The process begins with a low-resolution image x and a noisy image y<sub>t</sub> containing Gaussian noise, aiming to output a high-resolution image. This input undergoes T iterations of a specific formula to produce a super-resolution (SR) image. This procedure can be seen as progressively eliminating noise from the image. With sufficient iterations, noise is effectively removed, resulting in an enhanced super-resolution image.

![Unet]({{ '/assets/images/team3/unet.png' | relative_url }})
{: style=" max-width: 100%;"}
_Fig 5: Description of the U-Net architecture with skip connections. The low resolution input image x is interpolated to the target
high resolution, and concatenated with the noisy high resolution image yt. We show the activation dimensions for the example task of
16×16 → 128×128 super resolution._[4]

Spatial Super-Resolution:

Use bilinear interpolation to upsample a low-resolution video (such as 32 x 80 x 48) to a high-resolution video (such as 32 x 320 x 192) x, then concatenate x with the noisy high-resolution video y<sub>t</sub> in the channel dimension as the input data of U-Net.

Temporal Super-Resolution:

Upsample a low frame number video (such as 32 x 320 x 192) to a high frame number video (such as 64 x 320 x 192) x by inserting blank frames or repeated frames. Then concatenate x with the noisy high-frame video y<sub>t</sub> in the channel dimension as the input data of U-Net.

### V-Prediction

Instead of using conventional $$\epsilon$$-prediction to add noise, the author introduces a new technique known as velocity prediction parameterization, or v-prediction, for the video diffusion model. This method is summarized by the prediction formula $$v\equiv\alpha_t\epsilon-\sigma_tx$$, leading to $$\hat{x}=\alpha_tz_t-\sigma_t\hat{v}_\theta(z_t)$$.

![Vpred]({{ '/assets/images/team3/visualization_vpred.png' | relative_url }})
{: style=" width: 600px;max-width: 100%;"}
Fig 6: Visualization of reparameterizing the diffusion process in terms of and $$v$$ and $$v_\phi$$.\[4]

Let's delve into the derivation of the first equation. Remember the noise addition equation in DDPM:

$$
x_t=\sqrt{\bar{a}_t}x_0+\sqrt{1-\bar{a}_t}\epsilon \hspace{1cm}
(1)


$$

where the sum of the squares of the coefficients of $$x_0$$ and $$\epsilon$$ equals 1. This allows for a substitution using sine and cosine. The paper adopts the noise addition equation

$$
z_t=\alpha_tx+\sigma_t\epsilon \hspace{1cm}(2)
$$

Before proceeding with the derivation, it's important to align our symbols: in equation (1), $$x$$ represents $$x_0$$, while $$\alpha_t$$ and $$\sigma_t$$ correspond to $$\sqrt{\bar{a}_t}$$ and $$\sqrt{1-\bar{a}_t}$$, respectively. Also, $$z_t$$ in equation (2) represents $$x_t$$ in equation (1).

Defining $$\alpha_\alpha=\cos(\phi)$$ and $$\sigma_\alpha=\sin(\phi)$$, we obtain $$z_{\phi} =\cos(\phi)x+\sin(\phi)\epsilon$$. Taking the derivative of $$z$$ with respect to $$\phi$$, we find the velocity equation:

$$
v_{\phi} =\frac{d z_{\phi}}{d\phi}=\frac{d\cos(\phi)}{d\phi}x+\frac{d\sin(\phi)}{d\phi}\epsilon=\cos(\phi)\epsilon-\sin(\phi)x
=\alpha_{\phi}\epsilon-\sigma_{\phi}x \hspace{1cm} (3)
$$

Next, we apply $$\alpha_t$$ to both sides of equation (2) and $$\sigma_t$$ to both sides of equation (3), resulting in: $$\sigma v=\alpha \sigma \epsilon- \sigma^2 x$$, and $$\alpha z=\alpha^2 x+\alpha \sigma \epsilon$$. Subtracting these two equations cancels the term $$\alpha \sigma \epsilon$$, yielding: $$\sigma v-\alpha z=-\sigma^2 x-\alpha^2 x= -(\sigma^2+\alpha^2)x$$. Given that the sum of the squares of these two terms equals 1, we conclude with $$\hat{x}=\alpha_tz_t-\sigma_t\hat{v}_\theta(z_t)$$.

Remark: Similar to $$\epsilon$$-prediction, v-prediction introduces noise to the samples during the training process. However, when calculating the MSE loss function, velocity is used in place of $$\epsilon$$.

In the context of video diffusion model training, the author highlights the following advantages of v-prediction:

- It significantly enhances numerical stability throughout the diffusion process, facilitating progressive distillation.
- It prevents the temporal color shifting often observed in models using $\epsilon$-prediction.

![Vep]({{ '/assets/images/team3/v_ep_comp.png' | relative_url }})
{: style=" max-width: 100%;"}
_Fig 6: Comparison between $$\epsilon$$-prediction (middle row) and v-prediction (bottom row) for a
8×80×48→8×320×192 spatial super-resolution architecture at 200k training steps. The frames
from the $$\epsilon$$-prediction model are generally worse, suffering from unnatural global color shifts across
frames. The frames from the v-prediction model do not and are more consistent._[2]

```
if self.config.prediction_type == "epsilon":
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    pred_epsilon = model_output
elif self.config.prediction_type == "sample":
    pred_original_sample = model_output
    pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
elif self.config.prediction_type == "v_prediction":
    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
```

Sample results from Imagen

<div style="display: flex; justify-content: center;">
    <div style="flex: 50%; padding: 5px;">
        <video src="assets/images/team3/panda_car.mp4" alt="Image 1" style="width: 100%;" type="video/mp4" controls />
    </div>
    <div style="flex: 50%; padding: 5px;">
        <video src="assets/images/team3/bear.mp4" alt="Image 2" style="width: 100%;" type="video/mp4" controls />
    </div>
</div>

## Experiments

We experimented with a video diffusion models using this [diffusion model](https://huggingface.co/multimodalart/diffusers_text_to_video/blob/main/Text_to_Video_with_Diffusers.ipynb). We examine results when changing the number of inference steps below. All of the images use the same prompt, duration, and number of frames.

The prompt is 'A futuristic cityscape at dusk, with flying cars weaving between neon-lit skyscrapers'.

<div style="display: flex; justify-content: center;">
    <div style="flex: 33.33%; padding: 5px;">
        <video src="assets/images/team3/5steps.mp4" alt="Image 1" style="width: 100%;" type="video/mp4" controls />
    </div>
    <div style="flex: 33.33%; padding: 5px;">
        <video src="assets/images/team3/25steps.mp4" alt="Image 2" style="width: 100%;" type="video/mp4" controls />
    </div>
    <div style="flex: 33.33%; padding: 5px;">
        <video src="assets/images/team3/50steps.mp4" alt="Image 3" style="width: 100%;" type="video/mp4" controls />
    </div>
</div>

We see from these results that a higher number of inference steps leads to an image that more accurately reflects the prompt. With 5 steps, the video hardly reflects the prompt. With 25 steps, a resemblance is shown but the image is largely. With 50 steps, the image accurately shows the prompt.

## Reference

[1] Ho, Jonathan, et al. "Video diffusion models." arXiv:2204.03458 2022.

[2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models" arXiv:2210.02303 2022.

[3] Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. In ICLR, 2022.

[4] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad
Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 2022c.

[5] Brooks, Peebles, et al. Video generation models as world simulators, OpenAI, 2024

---
