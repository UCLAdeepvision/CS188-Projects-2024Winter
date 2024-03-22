---
layout: post
comments: true
title: Galaxy Morphology
author: Aaron Shi, Diana Estrada, Arturo Flores
date: 2024-03-20
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

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
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

### Bonus

##### Running the Existing Codebase
Notebook can be found here: https://colab.research.google.com/drive/1bE4KLPQApzIES9TNDv-r-FLwSINubxcx?usp=sharing 

We used Github’s provided quickstart guide to install Zoobot, download the datasets, use their custom GalaxyDataModule, and fine tune it on some data for a binary classification task. In particular, the data was the “Demo Ring Dataset” - a small dataset of only 1000 galaxies labeled as ring galaxies or non-ring galaxies. The results from this quickstart pipeline provided these labeled images showing the model’s predictions.

![Rings]({{ '/assets/images/team24/rings.png' }})
{: style="width: 400px; max-width: 100%;"}
*Fig 6. Labelled Ring Predictions: Example of what the quickstart guide outputs as labelled image predictions* [6].

##### Implementing Own Idea
Our second notebook can be found here: https://colab.research.google.com/drive/1Pn8xqEQ0UC1rXGIebt82LUhaafl2rWiA?usp=sharing 

In this notebook, we investigated how we could modify the quickstart guide in a meaningful way. Initially, we wondered if we could optimize the model in some way, like by changing the activation functions or the optimizer, but since that code was abstracted away by the GalaxyDataModule module of their Zoobot library, we instead chose to modify the quickstart by training on a different task on a different dataset. 

In particular, we trained on the GZ2 dataset which had many columns (only some are shown below), which was challenging to understand. GZ2 also has many more images at around ​​210K. 

![Rings]({{ '/assets/images/team24/cols.png' }})
{: style="width: 400px; max-width: 100%;"}
*Fig 7. GZ2 Column Names: A sample of the data columns of the GZ2 Dataset* [7].

Eventually though we realized that there was a “summary” column which we used as our multi-class classification target labels. 

![LabelCounts]({{ '/assets/images/team24/col_counts.png' }})
{: style="width: 400px; max-width: 100%;"}
*Fig 8. GZ2 Label Counts: The class counts across the entire GZ2 Dataset* [8].

After cleaning the data, dropping unnecessary columns, and modifying the model architecture to work for this new problem, we were able to plot predicted-labels for unseen images as below.

![MultiPredictions]({{ '/assets/images/team24/multi.png' }})
{: style="width: 400px; max-width: 100%;"}
*Fig 9. MultiClassification Task Output: Sample model output on unseen data* [9].

Lastly, we made a confusion matrix to better quantify the success of the model.

![ConfusionMatrix]({{ '/assets/images/team24/conf_matrix.png' }})
{: style="width: 400px; max-width: 100%;"}
*Fig 9. MultiClassification Confusion Matrix: Visual representation of multiclassification model effectiveness* [10].

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
