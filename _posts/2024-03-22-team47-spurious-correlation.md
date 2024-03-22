---
layout: post
comments: true
title: Mitigating Spurious Correlations in Deep Learning
author: Siddhartha Mishra, Pranav Varmaraja 
date: 2023-03-22
---

> Spurious correlations pose a major challenge in training deep learning models that can generalize well to arbitrary real life data. Models often exploit easy-to-learn features that are correlated with prediction(s) in the training set but not semantically meaningful in the real world. In this post, we provide an overview of the spurious correlation problem by characterizing it mathematically, discuss some of its causes, and describe three key methods for mitigating such correlations: GroupDRO, GEORGE, and SPARE, along with some empirical results for each method.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction to Spurious Correlations

Deep neural networks have achieved remarkable success across a wide range of tasks. However, they often rely on superficial correlations between input features and labels that do not reflect a meaningful relationship. For example, in a dataset of images of cows and camels, the model may learn to associate "cow" with green pastures and "camel" with sandy deserts, rather than learning the relevant semantic features of the animals themselves. As a result, the model may fail to generalize to cows in deserts or camels in pastures, an obvious bias. This sort of problem is known as a spurious correlation. In general, a model learns spurious correlations when it begins to associate certain features with prediction(s), when those features are not truly related to the prediction, i.e. in the camel/cow example.

## Deeper Characterization of Spurious Correlations

### Conventional Optimization Objective (ERM)
We begin by providing a mathematical overview of the conventional optimization objective used to train the models in question. The particular method used is known as Empirical Risk Minimization (ERM). The canonical equation for ERM is given below.

$$
w^* \in \text{arg min}_w \mathbb{E}_{(x_i,y_i) \in \mathcal{D}} [\ell(f(w, x_i), y_i)],
$$

Where each symbol is defined as follows.
- $$w^*$$: The optimal weights or parameters for our model. We attempt to find the best possible values for these weights that minimize the loss over our training dataset.
- $$\text{arg min}_w$$: The set of values $$w$$ which minimize the following expectation.
- $$\mathcal{D}$$: The data distribution from which our data points $$ (x_i, y_i) $$ are drawn. In practice, this is often the empirical distribution defined by our available dataset.
- $$\ell$$: A loss function that measures the discrepancy between the predicted values $$ f(w, x_i) $$ and the actual labels $$ y_i $$ (commonly a cross entropy loss in the case of multiple classification).
- $$f(w, x_i)$$: Our model's prediction, with weights $$ w $$ and input features $$ x_i $$.

The goal of the optimization problem described by this equation is to find the best parameters $$ w $$ that lead to the smallest possible average loss on our data. By doing so, we train a model that is, hopefully, both accurate and generalizable.

However, in the presence of spurious correlations, ERM may learn to rely on the simple spurious features to minimize the average loss. This results in poor worst-group performance, i.e., high error on the group where the spurious correlation does not hold. A few example cases of this are given in the next section.

### Examples of Spurious Correlations
<!-- insert image here -->
Some common examples of spurious correlations include:

Object recognition: The background (e.g., grassy, sandy, watery) may be spuriously correlated with the object class (e.g., cow, camel, waterbird), when the background is not truly a core feature determining the nature of the animal in the input image.

Natural language inference: The presence of certain words (e.g., "not", "no") may be spuriously correlated with the contradiction label, even when this might not be the case.

As a more concrete example, see:

![SpuCoEx1]({{ '/assets/images/47/spuco_ex1.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Examples of Spuriously Correlated Features in Two Separate Contexts* [TODO: FIX].

Consider the top row of this image. In the training dataset, we have a that images of females with blonde hair are common, as are images of males with dark hair. However, for example, images of males with blonde hair are rare. Thus, if we train a model to classify the gender of a person given an image of their face using the following dataset, the model will likely have significantly higher error rates on atypical groups (e.g. males with blonde hair here). This is an example of a spuriously correlated feature, where the core feature is male/female, and the spurious feature is the hair color (which we trivially know to not determine the gender of a person).

A similar situation is present in the second row of the image. In this case, we are concerned with classifying landbirds vs. waterbirds (core feature). Within the dataset, landbirds are often pictured with land (green, grassy, foliage) in the background, and waterbirds are often pictured with water in the background. However, we know that the background of the image does not determine what type of bird is necessarily present (we can have waterbirds on land and vice versa). Thus, a model trained on this dataset to classify the core feature of land/water bird may learn the spurious correlation between the background and the type of bird in the image. This may lead to lower performance on atypical groups (waterbirds on land, or landbirds in water).

This interpretation of a model potentially highly weighing a spurious feature when performing classification is also supported by a Grad-CAM heatmap of the model's outputs given various images:

![SpuCoEx2]({{ '/assets/images/47/spuco_ex2.png' | relative_url }})
{: style="width: 900px; max-width: 100%;"}
*Fig 2. Grad-CAM heatmaps of model(s) which have learned spurious features.* [TODO: FIX].

The above heatmaps support this intuition. We see much higher weights being placed on the backgrounds of the images vs the actual subject (which is the object to be classified in both cases). The fourth image from the left demonstrates the spurious correlation exactly. We see the model highly weighing most of the pixels in the background and explicitly not taking into account the region where the bird is in the image. This same phenomenon occurs in the other images as well, with the model placing a significant weight on the background (which we want to be relatively unrelated to model output). Thus, these activation mappings go to show the nature of a model which has learned spurious correlations.

### Spurious Correlation Key Metric (Worst-Group Test Error)

To address spurious correlations, the end goal is to minimize the worst-case test error:

$$
Err_{wg} = \max_{g \in \mathcal{G}} \mathbb{E}_{(x_i,y_i) \in g} [y_i \neq \hat{y}(w,x_i)]
$$

Where the symbols in the equation represent the following:
- $$Err_{wg}$$: The worst group error, a.k.a the highest error rate across all considered groups within the testing set.
- $$\mathcal{G}$$: A set of groups that we have partitioned our data into. This could be based on different subpopulations, spurious/core feature pairs, or other divisions relevant to the analysis.
- $$y_i \neq \hat{y}(w,x_i)$$: An indicator function that equals 1 if the prediction $$\hat{y}(w,x_i)$$ does not match the actual label $$y_i$$, and 0 if they match. Essentially, it tallies the misclassifications.

This metric is especially important when considering fairness and bias in machine learning models. By focusing on the worst-case error, we can understand how a model performs on the most challenging or underrepresented group in the dataset, which is often a critical aspect of model evaluation. This encourages the model to learn features that are predictive across all groups, rather than relying on spurious correlations.


## Causes of Spurious Correlations

There are several factors that cause deep learning models to be susceptible to spurious correlations:

Simplicity bias: Gradient descent-based optimization has a tendency to find simple functions that fit the training data before more complex ones [2]. Spuriously correlated features are often easier to learn than the true semantically meaningful features.
Limited training data: With small datasets, spurious correlations in the training set are more likely to be the most predictive features.
Imperfect inductive bias: While architectural inductive biases such as convolutions help, current neural network architectures are still not strong enough to ignore spurious features in favor of more generalizable ones.
## Methods for Mitigating Spurious Correlations

Researchers have proposed various approaches to overcome the problem of spurious correlations. We highlight three notable methods below.

### GroupDRO (Labeled Groups)

Group Distributionally Robust Optimization (GroupDRO) [3] is a method that assumes access to group labels at training time. The key idea is to optimize for the worst-case test set, by minimizing the maximum loss over predefined groups:

Intuitively, GroupDRO aims to learn a model that performs well on all groups, thus preventing it from relying on spurious correlations that only hold for some groups. The main limitation is that it requires knowledge of the groups at training time.


### GEORGE (Unlabeled Groups)

GEORGE (Group Robustness via Clustering) [4] is a method that does not require predefined group labels. The key idea is to:

Train a vanilla ERM model to learn an embedding that can separate the majority groups with the spurious features;
Cluster the learned embedding to identify the groups;
Train a robust model using GroupDRO with the inferred groups.
The intuition is that examples with the same spurious feature (e.g. grassy background) will be close in embedding space even if they belong to different classes, allowing them to be grouped via clustering.
GEORGE eliminates the need for human-specified group labels, but relies on the quality of the learned embedding to identify meaningful groups.

Thus, GEORGE essentially performs a similar process to GroupDRO, but it does not use predetermined groups and rather determines groups via unsupervised clustering before performing balanced sampling.

Using the `spuco` Python package maintained by the authors of [2] and [3], we can implement and evaluate a model trained via GEORGE. In this case, we evaluate the results of training via GEORGE on a simple LeNet. The high-level code to train a LeNet using GEORGE on the SpuCoMNIST dataset (a slight modification of MNIST to be suitable for SpuCo evals) is as follows.

First, initiallize the SpuCoMNIST dataset splits:

```python
import torch
# set torch device to use metal
# set to cuda if needed
DEVICE = torch.device("mps")
from spuco.utils import set_seed
set_seed(0)

from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty

# MNIST classes and spurious feature difficulty setting
classes = [[0,1], [2,3], [4,5], [6,7], [8,9]]
difficulty = SpuriousFeatureDifficulty.MAGNITUDE_LARGE

# initialize validation set from SpuCoMNIST
valset = SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes = classes,
    split="val",
    download=True
)
valset.initialize()

from spuco.robust_train import ERM
import torchvision.transforms as T

# initialize training set from SpuCoMNIST
trainset = SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    classes = classes,
    split="train",
    download=True
)
trainset.initialize()


# initialize testing set from SpuCoMNIST
testset = SpuCoMNIST(
    root="data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes = classes,
    split="test",
    download=True
)
testset.initialize()
```

Then, initialize a LeNet using the built in higher level model factory and set up a vanilla ERM
training pipeline for a single epoch using relatively standard optimization hyperparams (SGD with batch size = 64, lr=1e-2, momentum=0.9, nesterov).


```python
from spuco.models import model_factory
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(DEVICE)

from torch.optim import SGD

# NOTES: noticed some issue with dump pickler when using multiple workers
# for DataLoaders in both ERM and Evaluator classes

# not completely sure the cause (why mp pickling was broken on my system) 
# setting num_workers=0 rectified the issue, with slightly slower data loading
erm = ERM(
    model=model,
    num_epochs=1,
    trainset=trainset,
    batch_size=64,
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
    device=DEVICE,
    verbose=True,
)
erm.train()

```

As a baseline, we can evaluate the accuracy of the model here:
```python
# robust accuracy eval

from spuco.evaluate import Evaluator

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=DEVICE,
    verbose=True
)
evaluator.evaluate()
```
On our training run, we obtained an average accuracy of 99.494%, and a worst group accuracy of 0%, implying that in the worst group in the test set, not a single classification was correct. This worst group accuracy is using precomputed groups, which we only use for evaluation in this toy example using GEORGE. Normally, if such groups are available, GroupDRO will likely yield better results using the true groups vs inferred groups.

To fully train the model via GEORGE, we then perform the next step: clustering. That is, after we perform a single epoch of training using a standard ERM loss, we can compute groups using an unsupervised clusting algorithm on the output logits (embeddings) of the model.
```python
# we observe that although average accuracy over all groups is very high
# we see that within subgroups that accuracy is as low as 0

# --------------------------------------
# STEP 2: Cluster inputs into sublabels
# --------------------------------------
from spuco.group_inference import Cluster, ClusterAlg

logits = erm.trainer.get_trainset_outputs()
cluster = Cluster(
    Z=logits,
    class_labels=trainset.labels,
    cluster_alg=ClusterAlg.KMEANS,
    num_clusters=2,
    device=DEVICE,
    verbose=True
)
# partition train dataset into clusters using unsupervised
# NOTE: ran into same mp num_workers DataLoader issue, set num_workers=0 within get_trainset_outputs

group_partition = cluster.infer_groups()

```

Using these group partitions, we can then perform a balanced sampling of the inferred groups similar to GroupDRO while retraining the LeNet.

```python
# --------------------------------------
# STEP 3: retrain lenet using group balancing
# --------------------------------------

from spuco.robust_train import GroupBalanceBatchERM

balanced_model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(DEVICE)

group_balance_erm = GroupBalanceBatchERM(
    model=model, # could also perform another epoch(s) on previously ERM'd model
    num_epochs=4,
    trainset=trainset,
    group_partition=group_partition,
    batch_size=64,
    optimizer=SGD(balanced_model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True),
    device=DEVICE,
    verbose=True
)

group_balance_erm.train()

```

And lastly, we can evaluate the results of training a LeNet using GEORGE.

```python
# evaluate group balanced model
balanced_evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=balanced_model,
    device=DEVICE,
    verbose=True
)

balanced_evaluator.evaluate()
```
The evaluation yields a lower overall average accuracy of 93.85%, but a much better (considering the lightweight training run) worst group accuracy of 8.585%.

### SPARE (Unlabeled Groups)

SPARE (Mitigating Spurious Correlations by Resampling) [5] is another method that does not require group labels. The key observations are:

Due to the simplicity bias, spurious features are learned very early in training before the semantic features;
The model will be overly confident for examples containing majority group spurious features.
Based on these observations, SPARE:

Trains the model for only a few epochs and clusters the model predictions within each class to identify majority/minority groups;
Upweights minority groups and downweights majority groups during retraining, to prevent the model from relying on spurious features.
Like GEORGE, SPARE infers groups in a data-driven way. However, it does so very early in training to capture spurious features before semantic ones. It also directly reweights examples to avoid spurious features rather than relying on the worst-case loss.

## Tradeoffs when Dealing with Spurious Correlations

There are several key tradeoffs to consider when dealing with spurious correlations:

Robustness vs accuracy: Methods that improve worst-group performance often do so at the cost of average performance. This is because the model is forced to use more complex features rather than relying on simplistic spurious correlations. The right balance depends on the application - for high-stakes applications like medicine, robustness may be prioritized over accuracy.
Labeled vs unlabeled groups: Methods like GroupDRO require the groups to be predefined, which provides stronger control over the model's behavior. However, manually specifying groups can be challenging and time-consuming. Methods like GEORGE and SPARE infer the groups automatically, but may identify groups that don't perfectly align with the spurious features.
Early vs late intervention: SPARE intervenes very early in training to identify and mitigate spurious features. This prevents the model from overfitting to spurious features in the first place. In contrast, GEORGE allows the initial model to rely on spurious features, but then uses them to infer groups for robust training. The optimal point of intervention likely depends on the strength and granularity of the spurious features.
Worst-case vs reweighted loss: GroupDRO optimizes for worst-case group performance, while SPARE reweights examples to balance group performance. Worst-case optimization provides a strong guarantee, but may be overly pessimistic. Reweighting is a softer approach, but requires careful tuning of the weights.
Ultimately, the best approach depends on the nature of the spurious correlations, the availability of group labels, and the specific robustness and accuracy requirements of the application.

## Conclusion

Spurious correlations are a key challenge in training models that can generalize to out-of-distribution data. While methods like GroupDRO are effective when group labels are available, GEORGE and SPARE show promising results in automatically identifying and mitigating spurious correlations.

Looking ahead, there are several important directions for future work:

Developing architectures and regularizers that have a stronger inductive bias towards semantic features over spurious ones.
Improving methods for automated discovery of spurious features, e.g., through unsupervised learning or causal reasoning.
Characterizing the tradeoffs between robustness and accuracy in the presence of spurious correlations, and developing methods to navigate these tradeoffs.
Extending these methods to more complex data modalities and tasks, such as video understanding and language generation.
Continued research along these lines will be crucial for building more robust and generalizable machine learning systems.

## Reference

[1] Sohoni, Nimit S., et al. "No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems." arXiv:2011.12945v2 [cs.LG]. 2022.

[2] Joshi, Siddharth, et al. "Towards Mitigating Spurious Correlations in the Wild: A Benchmark & a more Realistic Dataset." arXiv:2306.11957v2 [cs.LG]. 2023.

[3] Yang, Yu, et al. "Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias" arXiv:2305.18761v2 [cs.LG]. 2024.

---
