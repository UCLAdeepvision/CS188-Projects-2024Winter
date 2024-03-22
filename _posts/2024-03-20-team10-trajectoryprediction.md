---
layout: post
comments: true
title: Trajectory Prediction
author: Yu-Chen Lung, Edward Ng, Warrick He, Alan Yu
date: 2024-03-22
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Alan inserts his introduction to the problem here. Can base off of past ones.

## Papers and Approaches

# Social LSTMs

[Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)

Humans have the ability to “read” one another, predicting each other’s general motions, each obeying a set of untold rules that help us predict the motion of other humans. However, trying to predict this motion with machines has proven to be a difficult task. Previous methods have either used functions to predict behavior or only consider other people in close proximity.

> Social LSTMs tackle the problem of trajectory prediction, more specifically pedestrian trajectories, and allow for anticipating more distant interactions.

#### Social LSTM Architecture

Social LSTMs base themselves on the traditional LSTM architecture, which has proven to be very useful for sequence prediction tasks such as caption generation, translation, video translation, and more. In their specific model, the authors of Social LSTM introduce a few key ideas. Firstly, each trajectory is represented by its own LSTM model. Secondly, the LSTMs are linked to each other via a special layer called a social pooling layer.

![SocialLSTM]({{ '/assets/images/team10/sociallstms.png' | relative_url }})
{: style="width: 400px; max-width: 100%; text-align: center;"}

## Social GANs

[Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/pdf/1803.10892.pdf)

![Social GANs]({{ '/assets/images/team10/socialGAN.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

### Introduction
Human motion is inherently ‘multi-modal’: given any position, we have many ‘socially plausible’ methods of maneuvering throughout space to get to our destination. This suggests modeling a recurrent sequence to make predictions on future behavior and pooling information across various agents that impact possible suggested movements. Previous methods, RNN-based architectures and traditionally used methods based on specialized features (i.e. physics-based approaches) suffered from certain primary limitations: using only a neighborhood of agents to compute due to being unable to model interactions between all agents in a computationally efficient manner (1) and only being able to learn ‘average behavior’ instead of generally acceptable and more likely behaviors based on a history of movements at a given time step (2). 

### Social GAN Architecture
This following proposed architecture addresses these limitations by employing Generative Adversarial Networks which have empirically been able to overcome intractable probabilistic computations and further projections of behaviors. Instead of generating images, as has been commonly employed in the past, the architecture proposes to generate multiple trajectories given a history where a generator creates candidates and the discriminator evaluates them, pushing the generator to learn ‘good behaviors’ that can fool the discriminator (to generate ‘socially acceptable motion trajectories’) in a crowded environment with numerous agents. 

![Social GANs]({{ '/assets/images/team10/sgarch.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

The model: an RNN Encoder-Decoder generator and RNN-based encoder discriminator with variety loss to encourage the generator to explore various plausible paths while being consistent with provided inputs, combined with global pooling to encode information with respect to all agents in a given scene. In the provided figure, there are several key components, the initial generator ($$G$$), pooling module, and discriminator ($$D$$). $$G$$ takes in previous trajectories $$X_i$$ and encodes the history of individual $$i$$ as $$H_i^t$$ where $$t$$ indicates the current time. The pooling module takes input the hidden state of the currently observed hidden state for each individual and outputs a pooled vector $$P_i$$ for each person. The decoder then produces the future trajectory conditioned on the latest observed hidden state $$H_i^{t_{obs}}$$ and $$P_i$$. $$D$$ takes in as input whether or not the trajectory is real or fake and classifies them as socially acceptable or not. This allows for the generator to converge on a predicted series of probabilities for given agents within a scene. Given all agents in a scene $$X = X_1, X_2, …, X_n$$, the goal is to predict all future trajectories simultaneously such that the input trajectory of person i is defined as $$X_i = (x_i^t, y_i^t)$$ for a $$t=1,...t_{obs}$$  and the future trajectory is defined similarly with $$Y_i = (x_i^t, y_i^t)$$ from time steps $$t=t_{obs} + 1,...,t_{pred}$$.

#### GANs
The generative adversarial network functions with two neural networks trained in complete opposition where generative model $$G$$ attempts to use the probabilistic data distribution to ‘fool’ the discriminator and discriminative model $$D$$ that estimates the probability that a provided sample was from $$G$$ versus the training set. $$G$$ inputs a latent variable, $$z$$, and outputs a generated sample $$G(z)$$; $$D$$ takes in a sample $$x$$ and outputs whether or not the input belongs to the data set or not. Thus we have the following ‘min-max objective’:

![Social GANs]({{ '/assets/images/team10/ganloss.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

where we maximize the capability of the discriminator and minimize $$G$$ to become close to generative distribution of the data such that the discriminator cannot tell the difference.

#### Socially-Aware GANs
The model consists of the aforementioned 3 key components: Generator ($$G$$), Pooling Module ($$PM$$) and Discriminator ($$D$$); $$G$$ takes inputs of agents $$X_i$$ and outputs predicted trajectories of said agents $$\hat{Y}_i$$. $$D$$ takes as input the entire sequence from the input movement trajectory and future predictions ($$X_i$$, $$\hat{Y}_i$$, respectively) and classifies whether or not each provided sequence/trajectory prediction pair is fake or real.

With the generator, using $$\phi(\cdot)$$ as a single layer MLP with ReLU non-linear activation, $$W_{ee}$$ as the embedding weight, $$W_{encoder}$$ as the shared embedding weight between all individuals, and the given locational input $$x_i^t, y_i^t$$, we obtain the following recurrence for the embeddings of each agent at time t for the LSTM cell of the encoder:

![Social GANs]({{ '/assets/images/team10/socialganeq2.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

However, one LSTM per person fails to capture the understanding of inter-agent interactions, so this is modeled via the Pooling Module such that each hidden state for all agents interact, giving a pooled hidden state $$P_i$$ for each person i.

To produce future scenarios with respect to the past, the generation of new output requires a conditional for the decoder initialization:

![Social GANs]({{ '/assets/images/team10/socialganeq3.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

where $$\gamma(\cdot)$$ is a fully-connected layer with ReLU activation and $$W_c$$ as the weight embedding.

After the given encoder and decoder initialization, we obtain the predictions as follows:

![Social GANs]({{ '/assets/images/team10/socialganeq4.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

The discriminator ultimately consists of a separate encoder which takes as input some real or generated data and classifies them accordingly. Based on the encoder’s last hidden state, the model uses an additional fully-connected layer to generate a classification score. This ideally teaches subtle social interaction rules and classifies trajectories that aren’t acceptable as ‘fake’, using L2 loss on top of adversarial loss to indicate how ‘far’ generated samples are from the ideal data distribution. To further encourage diverse samples on trajectory prediction (in the case of being able to move in multiple directions to mimic humans’ multi-modal movement capability), the authors employed a variety loss function which selected the best of $$k$$ generated output predictions by randomly sampling the latent space!

![Social GANs]({{ '/assets/images/team10/variety_loss.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

This loss considers only the best trajectory, pushing the network to cover the space that best conforms to the past trajectory and provides high probabilistic predictions in realistic directions.

![Social GANs]({{ '/assets/images/team10/pm_vs_social_pool.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

### Discussion 
Using the LSTMs, the authors were able to determine the hidden states with respect to each individual agent, but this additionally requires a mechanism to share information across LSTMs (i.e. between individuals). This is directly enabled with the pooling module in computing a resultant hidden state using the hidden states of all LSTMs for each agent; the module handles primary issues when scaling to variable numbers of people across a large crowd and handling human-to-human interactions that may be significantly more scarce environment (where agents further away may still impact one another, albeit to a lesser degree). With other social pooling mechanisms, they fundamentally work by employing a grid-based pooling scheme which fails to capture global extent as it only considers within the grid whereas the Pooling Module computes relative positions between each agent and all other agents which are concatenated with the person’s hidden state and processed independently via MLP. This allows the final pooled vector $$P_i$$ to summarize all the information a person needs to make a decision to ‘move’. 

When removing the pooling, we observe that agents fail to consider the paths of others, which in turn may result in convergent pathways (which are realistically infeasible when considering walking scenarios). Pooling via relative distance enforces a global understanding of other agents and what is technically feasible with respect to the current state of agents and the ‘conformity to social norms’ and practices when referring to movement. Observe the following image where we have agents without the pooling mechanism (SGAN) versus agents with the pooling mechanism (SGAN-P).

![Social GANs]({{ '/assets/images/team10/sgan_vs_sganp.png' | relative_url }})
{: style="width: 100%; max-width: 100%; text-align: center;"}

This directly mimics human behavior in representing those tending towards the same direction, depicting human tendency to vary pace in order to avoid collision (scenario 3) and even yield right-of-way (scenario 4). And, this represents the tendency to react to incoming movement as well (particularly in group behavior). Using information with respect to the pace of the person and the particular direction of movement (gathered from the initial time-steps), the paper observed different ‘reactions’ of predictions to the movement of crowds.

Another key observation was made relative to the latent space: in understanding the latent space $$z$$, particular directions within the latent space were directly associated with the direction and speed of the agents that served as the future predictions in upcoming time steps.

Overall, this methodology proposes a novel solution to generating predictions based on global interactions via a relative-distance pooling method, allowing the network to learn social norms of agent-movement by training against a discriminator. In directly imposing a variety loss with said pooling layers, this allowed training of a network that could produce multiple viable probabilistic solutions to trajectory predictions given diverse samples.


# VectorNet

# Other

## Comparisons
<!-- 
## Main Content
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
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/). -->

## Reference
Please make sure to cite properly in your work, for example:

[1] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei and S. Savarese, "Social LSTM: Human Trajectory Prediction in Crowded Spaces," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 961-971, doi: 10.1109/CVPR.2016.110.
[2] Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., & Alahi, A. (2018). Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1803.10892
---
