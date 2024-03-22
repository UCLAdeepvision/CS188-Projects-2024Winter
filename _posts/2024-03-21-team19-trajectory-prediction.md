---
layout: post
comments: true
title: Trajectory Prediction
author: Nevin Liang, Jeffrey Shen, Manav Gandhi
date: 2024-03-21
---


> In the AV pipeline, trajectory prediction plays a crucial role in the Autonomous Vehicle (AV) pipeline. In this report we’ll explore two different machine learning approaches to trajectory prediction for Autonomous Vehicles: a Conv-based architecture which uses rasterized semantic maps, and a GNN-based architecture which uses a vector-based representation of the scene. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Trajectory prediction is the process of predicting the future positions of agents over time. For Autonomous Vehicles, having an accurate trajectory prediction algorithm is incredibly important as it allows AVs to predict the movement of road-agents (i.e. other vehicles, pedestrians, etc.), and adjust its route accordingly to avoid accidents. Trajectory prediction is difficult because it often requires a holistic understanding of the scene, and driver behavior can be unpredictable. 

## ConvNet

A common way to tackle the problem is to encode the scene as a rasterized HD semantic map. This is typically done by taking the classified actors extracted from the AV’s perception system and overlaying their locations and class attributes on top of a bird’s eye view of the scene. This lets us treat the problem as a computer vision task, as we can then feed these maps into a Convnet-based architecture like ResNet. One advantage of this approach is that this allows for the use of various pretrained vision models, where the backbone can be used as a feature extractor.

### fwef

![Convnet]({{ '/assets/images/19/convnet.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. A deep learning-based approach to trajectory prediction* [1].

Instead of class predictions, the model returns $$n$$ pairs of values, where each pair represents the predicted future (x,y) location of a target, and $n$ represents the number of future frames to predict. The loss is simply calculated to be the Mean Squared Error (MSE) between the locations of the predicted and actual values of agents:

$$
L(x, x_p) = \frac{1}{n} \sum_{i}^{n} (x - x_p)^2
$$

where $$x$$ and $$x_p$$ can be either the x or y coordinate of each agent location at a given frame.
### Implementation

To see this model in action, we ran Woven Planet’s prediction [notebook](https://github.com/woven-planet/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb), which uses the Lyft’s Level 5 Self-driving motion prediction dataset [2] for training. While Woven’s notebook uses a pretrained ResNet model as the backbone, we decided to experiment using an EfficientNet instead. EfficientNet utilizes a technique called compound-coefficient to scale up models efficiently [3], which is especially useful given the scale of Lyft's dataset. The following code snippet shows how the EfficientNet backbone is implemented. Note how the first and last layer is modified in order to accommodate the input and output shape:

```
class EfficientNet(nn.Module):

    def __init__(self):
        super().__init__()

        # replace first and last layer to match shape of data
        self.model = efficientnet_b0(pretrained=True)
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        in_channels = 3 + num_history_channels
        self.model.features[0] = Conv2dNormActivation(
            in_channels,
            self.model.features[0].out_channels,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU
        )
        # num future states * x, y coordinates
        num_targets = 2 * cfg["model_params"]['future_num_frames']
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_targets)

    def forward(self, x):
        x = self.model(x)
        return x

    def to(self, device):
        return self.model.to(device=device)
```

### Results

![Resnet and Efficientnet Results]({{ '/assets/images/19/convnet_results.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Visualizations of Resnet and Efficientnet performance during training. The cyan dots represents the model's predicted location, while the purple dots represents the ground truth locations*.

From the visualizations above, we see that the Resnet and Efficientnet models are both able to predict the general path of each agent. However, one important thing to note is that Efficientnet was able to train the same number of iterations as Resnet in about 1/3rd of the time.

## VectorNet

One drawback that CNN-based architectures have is their relatively high memory and performance requirements, which is a key factor for Autonomous Vehicle systems as they require real-time calculations. VectorNet was designed to overcome those issues. The core difference between previous trajectory prediction models and VectorNet is how the model represents the scene: instead of a rasterized image, VectorNet treats the scene as a composition of vectors, where each vector can represent either a trajectory or a component of 

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

[2] Houston, John, et al. "One Thousand and One Hours: Self-driving Motion Prediction Dataset." *CoRR*. 2020.

[3] Tan, Mingxing & Le, Quoc. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *International Conference on Machine Learning*. 2019.

[4] Gao, Jiyang, et al. "VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation." *Conference on Computer Vision and Pattern Recognition*. 2020.

---
