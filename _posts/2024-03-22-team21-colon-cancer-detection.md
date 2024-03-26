# Polyp Segmentation for Colorectal Cancer

# Introduction

Colon cancer is the third most commonly diagnosed cancer in the world, with nearly one million deaths in the year 2020 alone. It occurs when abnormal tissues called polyps in the color become malignant tumors. A colonoscopy is a procedure that visually inspects the colon lining for polyps. Doctors use colonoscopies as the gold standard to screen for colon cancer, which has led to a 30% decline in incidences as studies have shown. However, in the colonoscopy process, it can be hard to distinguish these abnormalities. The colon polyp miss rate could range from 20 to 47%. To cover this shortcoming, medical professionals have started to utilize medical image segmentation. In the case of colon cancer, detecting polyps using medical image segmentation requires algorithms that can detect polyps that are of variable shapes, sizes, and low-intensity contrast. Medical image segmentation involves dividing digital medical images into distinct regions to isolate specific structures or areas of interest within the image. Deep learning algorithms, particularly convolutional neural networks, help automate and improve the segmentation process.

To segment colons, CT scans are used to image the colon area during colonoscopy. CT scans consist of multiple slices of of different views of an image, stacked together to form a 3D image. Previous techniques used to segment the colons include morphological operations, region growing and classical machine-learning. Morphological operators modify the shapes and boundaries of colon structures, while region growing algorithms expand regions of interest based on intensity levels, aiding in the precise delineation of colonic tissues and abnormalities. However these techniques struggled with segmenting various region-sizes and backgrounds. Classical machine-learning segmentation models required hand-crafted features, which lacked performance sufficient for clinical practice. Most segmentation models use newer technologies, and follow a UNet-like architecture, with Convolutional Neural Networks (CNNs) commonly used in the encoder. Some transformer models specifically used are TransUnet, SwinUnet, SegTran, and TransFuse. However, CNNs, while effective in segmentation tasks, have limitations. They excel at capturing local information but struggle to grasp global context. Additionally, many encoder-decoder segmentation models hinder multi-scale feature fusion by utilizing a bottom-up aggregation path in the decoder. Overall, the intricacies in applying deep learning for polyp detection make it a challenging yet crucial area of research, given its significant practical applications in medical imaging and diagnosis.

![*Figure 1: Different Possible Polyps In Need of Segmentation [1].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled.png)

*Figure 1: Different Possible Polyps In Need of Segmentation [1].*

# Approach 1: [Meta](https://ieeexplore.ieee.org/document/10179485)

### Motivation

The motivation behind the Meta approach is to address the challenges of polyp segmentation in medical imaging, particularly focusing on achieving high accuracy while maintaining efficiency for real-time application during colonoscopy procedures. Traditional methods, including CNNs and transformer-based approaches, have limitations in handling fine-detailed features, computational cost, and real-time requirements. The Meta approach aims to overcome these limitations by introducing a novel multi-scale efficient transformer attention mechanism integrated into the Unet architecture.

### Architecture

The architecture of the implemented model, known as META-Unet, is a convolutional neural network (CNN) with an underlying backbone architecture based on ResNet-34, which is pre-trained on a large dataset. The backbone extracts hierarchical features from the input images. These features are then processed by a series of modules, including multiscale efficient transformer attention (META) modules and segmentation heads.

The META module operates on multiple scales of feature maps simultaneously. It consists of two branches: a global branch and a local branch. The global branch utilizes self-attention mechanisms to capture long-range dependencies across the entire feature map, while the local branch employs self-attention within smaller local regions. This enables the model to capture both global context and fine-grained details effectively. Each META module consists of efficient transformer blocks, which utilize multi-head self-attention and feed-forward networks to process the feature maps.

The segmentation head takes the processed feature maps from the META modules and performs upsampling to generate pixel-wise segmentation masks. It consists of convolutional layers followed by bilinear upsampling to increase the spatial resolution of the feature maps. The final output of the model is a segmentation mask, where each pixel represents the predicted class of the corresponding input pixel.

![*Figure 2. Meta Architecture [1].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%201.png)

*Figure 2. Meta Architecture [1].*

### Code Discussion

The code implements the META-Unet architecture for polyp segmentation, featuring several crucial components. It utilizes classes such as CBR for 2D convolutional layers with batch normalization and PReLU activation, and Mlp for Feed Forward Networks with customizable hidden layers and activation functions. Additionally, the Self_Attention and Self_Attention_local classes handle multi-head self-attention operations, crucial for feature interaction across spatial dimensions. These components are integrated into the Efficient Transformer blocks within the META module, facilitating efficient feature transformation and aggregation. Lastly, the Seg_head class represents the segmentation head responsible for generating final segmentation masks, utilizing convolutional layers for upsampling and prediction. Overall, the META_Unet class effectively combines backbone feature extraction with attention mechanisms and segmentation head, offering methods for both feature extraction and segmentation prediction to achieve accurate polyp segmentation.

### Results

The experimental results demonstrate the effectiveness of the Meta approach in polyp segmentation across various datasets. With a Dice coefficient exceeding 0.9 on the HarDNet-MSEG dataset, the Meta approach achieves exceptional segmentation accuracy. Moreover, it consistently outperforms other methods, achieving an IOU of 0.86 on the Kvasir-SEG dataset. On the ETIS-Larib dataset with higher resolution images, the Meta approach outperforms other Transformer-based methods, demonstrating its effectiveness in handling small polyps and various segmentation challenges. Overall, the Meta approach offers a significant improvement in segmentation accuracy and generalization ability, showcasing its potential impact on enhancing the diagnosis and treatment of colorectal cancer.

![*Table 1. Meta Results [1].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%202.png)

*Table 1. Meta Results [1].*

### Conclusion/Summary

In conclusion, the Meta approach, META-Unet, presents a novel solution for polyp segmentation in medical imaging. By integrating a multi-scale efficient transformer attention mechanism into the Unet architecture, the method achieves high accuracy and efficiency, addressing the challenges of fine-detailed feature modeling, computational cost, and real-time requirements. The results suggest the potential clinical impact of the Meta approach in improving the diagnosis and treatment of colorectal cancer through more accurate and efficient polyp segmentation.

# Approach 2: [RaBit](https://arxiv.org/abs/2307.06420)

### Motivation

Another approach that addresses current problems in regard to the accuracy of polyp segmentation from CT scans is the RaBit network. It addresses the limitations of existing deep networks concerning the representation of multi-scale features and generalization capabilities. The RaBit network's improvement lies in the incorporation of a lightweight transformer model, featuring a bidirectional pyramid decoder with reverse attention. 

The motivation for the raBiT network stems from the limitations of existing segmentation models, particularly those using UNet-like architectures. While Convolutional Neural Networks (CNNs) have shown impressive performance, they struggle with capturing global context information and multi-scale feature fusion, crucial for efficient semantic segmentation. Additonally, many encoder-decoder models restrict multi-scale feature fusion, hindering semantic segmentation efficiency.  Attention mechanisms are widely used for computational efficiency while maintaining focus on relevant information. Recent interest in Transformers for semantic segmentation, seen in models like TransUNet and TransFuse, leverages self-attention layers. However, TransUNet's computational demands raise concerns. raBiT seeks a balanced approach, combining Transformer strengths for global context with an efficient architecture, addressing limitations in existing models.

### Architecture

RaBiT, a Transformer-based network tailored for polyp segmentation, integrates a hierarchically structured lightweight Transformer encoder and a bidirectional feature pyramid network decoder. RaBiT addresses limitations in capturing multi-scale and multi-level features by proposing a lightweight binary Reverse Attention (RA) module with bottleneck layers, iteratively refining polyp boundaries. Additionally, a multi-class RA module is introduced that meets the needs for multi-class segmentation.

![*Figure 3. RaBit Architecture [2].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%203.png)

*Figure 3. RaBit Architecture [2].*

**Encoder**

RaBit uses a encoder called a mix transformer(miT) which is designed to generate CNN like multi-scale featues. Specifically, MiT-b3 is used as the backbone transformer for the architecture. It consists of five feature levels denoted as P1 to P5. Two new feature levels, P6 and P7, are derived from P5. To generate P6, the feature map P5 undergoes a transformation involving a 1x1 convolutional layer with 224 neurons and stride 1, followed by batch normalization and a 3x3 max pooling layer with stride 2. Subsequently, a 3x3 max pooling layer is applied to P6 to produce the feature map P7. The feature maps from P3 to P7 are directed to the decoder.

**Decoder**

A reverse attention module is utilized in a bottom-up direction to enhance polyp boundaries, starting from coarse to finer feature maps. This bottom-up approach involves refining details and boundaries by progressively analyzing the information at lower levels. Inspired by BiFPN, the refinement process iterates by fusing information between feature levels in a top-down direction. In top-down processing, higher-level cognitive processes guide the refinement of information at lower levels. The combination of top-down feature aggregation and bottom-up boundary refinement is termed RaBiFPN module, which can be repeated multiple times to form a stacked block. Fast normalized fusion is used to aggregate all feature maps in the decoder, with fusion weights learned during training via back-propagation, ensuring effective integration of information from various levels.

The introduced Reverse Attention (RA) module enhances the focus on object boundaries, particularly beneficial for segmentation tasks like polyp segmentation. RA channel-wise processes each feature map, emphasizing information related to object boundaries. The reverse operation refines this boundary information, and the enhanced details are integrated back into the original feature maps.

A modified RA module is introduced to address limitations of the original RA module, working with feature maps of size H × W × C, where C = 224. Before entering the decoder, a 3x3 convolution layer compresses higher resolution feature maps from P3 to P5 to match C = 224. Feature maps P6 and P7, already with C channels, go directly to the decoder. This modification improves adaptability for multi-class segmentation and preserves information during feature fusion.

For binary segmentation, used to detect polyps from the colon background, a 3x3 convolution compresses the input to a single channel, followed by a sigmoid function for RA. In multi-class segmentation, used to detect different types of polyps( benign and neoplastic), the input is compressed to n channels with a 3x3 convolution layer. A softmax function generates attention maps, which, when reversed and multiplied with the input, form concatenated feature maps.

**Loss**

For binary segmentation, a compound loss, combining weighted focal loss and weighted IoU loss is used. For multi-class polyp segmentation, the categorical cross-entropy loss is used. Deep supervision is used to multi-scale outputs to train the network,

### Results

The model was trained and applied to multiple datasets, including the CVC-ClinicDB and Kvsair for polyp segmentation. It was also applied to multi-class segmentation problems, including NeoPolyp-Large and -Small. These results display the RaBit approaches outcome using the Dice, IoU, Recall, and Precision metrics, compated to other previous models used for polyp segmentation. RaBit successfully improves on the scores for each of these previous models. 

![*Table 2. RaBit Binary Class Results [2].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%204.png)

*Table 2. RaBit Binary Class Results [2].*

![*Table 3. RaBit Multi-Class Results [2].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%205.png)

*Table 3. RaBit Multi-Class Results [2].*

![*Figure 4. RaBit Segmentation Results Images [2].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Untitled%206.png)

*Figure 4. RaBit Segmentation Results Images [2].*

### Conclusion/Summary

In conclusion, RaBit introduces a novel approach for polyp segmentation that implements a lightweight transformer as the encoder and utilizes multiple bidirectional feature modules and repeated reverse attention modules in the decoder. RaBit surpasses existing approaches on various benchmark datasets for polyp segmentation, proving its importance to the future of detecting colorectal cancer.

# Approach 3: [2D/3D](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6374810/pdf/JHE2019-1075434.pdf)

## Motivation

The motivation for the paper is to address the need for an automated and accurate method to segment colorectal tumors in 3D T2-weighted (T2w) MRI1. Nowadays, magnetic resonance imaging (MRI) is the most preferable medical imaging modality in primary colorectal cancer diagnosis for radiotherapy treatment planning. Manual segmentation of tumors in MRI scans is a time-consuming and challenging task that requires expert knowledge. This can lead to delays in diagnosis and treatment, Moreover, existing automated methods may not be accurate enough, potentially leading to incorrect diagnoses. This is a significant problem because early and accurate detection of colorectal tumors can significantly improve the prognosis and survival rates for patients. The limitations of 2D CNNs in handling volumetric spatial information (like 3D MRI scans) have led to the development of 3D CNNs, such as the 3D U-net, VoxResNet, and DenseNet. However, issues like the abundance of parameters and the potential for simplification arise. While previous methods like atlas-based and supervoxel clustering show promise, recent advancements in deep learning-based approaches, specifically 3D CNNs, have demonstrated impressive results. The proposed 3D MSDenseNet addresses these concerns by employing a multiscale training scheme with parallel 3D convolutional layers. Thus maintaining local and global contextual information throughout the network, improving segmentation efficiency.

## Architecture:

### Overview:

DenseNet, introduced by Huang et al., extends the skip connection concept by creating a direct connection from every layer to its corresponding previous layers. This ensures maximum gradient flow between layers. In DenseNet, feature maps from the preceding layer are concatenated and used as input to the subsequent layer, forming a dense connection structure. 

Yu et al. further extended DenseNet to three-dimensional volumetric data, introducing the densely connected volumetric convolutional neural network (DenseVoxNet). This 3D version is applied to volumetric cardiac segmentation, showcasing the effectiveness of DenseNet's dense connections in handling spatial information in three dimensions.

The 3D MSDenseNet introduces a novel approach by employing two interconnected levels, the depth level and the scaled level, for simultaneous computation of high- and low-level features. The input volume undergoes processing through subsequent layers (l) at different scales (s1 and s2).

In the initial layer, the feature map from the first convolutional layer is divided into the coarser scale (s2) through pooling with a stride of power 2. High-resolution feature maps in the horizontal path (s1) are densely connected in subsequent layers, while the output feature maps in the vertical path (s2) result from the concatenation of transformed features from previous layers in s2 and downsampled features from previous layers of s1, propagated diagonally.

This helps the network understand both the fine details and the overall context of the image as it processes information layer by layer.

### Specific Architecture

![*Figure 5. MSDenseNet Architecture [3].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Screenshot_2024-03-08_at_11.26.55_PM.png)

*Figure 5. MSDenseNet Architecture [3].*

The proposed CNN architecture is based on a 3D Multiscale Densely Connected Convolutional Neural Network (3D MSDenseNet). 

The network architecture follows a 3D end-to-end training approach with dual parallel paths: the depth path and the scaled path. The depth path includes eight transformation layers, while the scaled path has nine transformation layers. Each transformation layer in both paths comprises a Batch Normalization (BN) layer, followed by a Rectified Linear Unit (ReLU), and then a 3x3x3 convolution (Conv) layer, similar to the architecture of DenseVoxNet. Additionally, a 3D upsampling block, also inspired by DenseVoxNet, is incorporated into each path.

To enhance robustness and prevent overfitting, the network utilizes a dropout layer with a dropout rate of 0.2 after each convolutional layer. The proposed method has approximately 0.7 million total parameters, significantly fewer than DenseVoxNet (1.8 million) and 3D U-net (19.0 million). The implementation is carried out in the Caffe library.

### 3D Level Set

To enhance the precision of tumor boundary predictions from the discussed networks, a 3D level-set based on the geodesic active contour method is employed as a postprocessor. This algorithm refines the final predictions by adjusting tumor boundaries through the incorporation of prior information and a smoothing function.

The 3D level-set method establishes a relationship between the computation of geodesic distance curves and active contours. This relationship ensures accurate boundary detection even in the presence of significant gradient variations and gaps. 

## Results

![*Table 5. MSDenseNet Results [3].*](Polyp%20Segmentation%20for%20Colorectal%20Cancer%202e28f2c1fbcf41e685655503855cf9fa/Screenshot_2024-03-09_at_12.58.44_AM.png)

*Table 5. MSDenseNet Results [3].*

The method was evaluated using T2-weighted 3D MRI data from 43 patients diagnosed with locally advanced colorectal tumors (cT3/T4).

### Quantitative Results

Quantitative results were obtained by calculating the mean and standard deviation of each performance metric across 13 test volumes. Initially, a comparison was made without postprocessing using the 3D level set, considered as baseline methods. 

Subsequently, the comparison was presented with the incorporation of the 3D level set as a postprocessor to refine segmented boundaries. 

Eight settings were examined: 

1. 3D FCNNs, 
2. 3D U-net, 
3. DenseVoxNet, 
4. 3D MSDenseNet, 

and their counterparts with the 3D level set as a postprocessor.

Performance Metrics:

- Dice Similarity Coefficient (DSC): Measures overlap between predicted and ground truth segmentations.
- Recall Rate (RR): Indicates sensitivity in detecting tumor regions.
- Average Surface Distance (ASD): Quantifies spatial differences between predicted and ground truth contours.

Results for 3D MSDenseNet (mean ± standard deviation):

- Before postprocessing:
    - DSC: 0.8406 ± 0.0191
    - RR: 0.8513 ± 0.0201
    - ASD: 2.6407 ± 2.7975
- After postprocessing:
    - DSC: 0.8585 ± 0.0184
    - RR: 0.8719 ± 0.0195
    - ASD: 2.5401 ± 2.402

### Analysis:

Results in Table 1 show that 3D FCNNs had the lowest performance across all metrics, followed by 3D U-net and DenseVoxNet. The proposed method demonstrated the highest values for Dice Similarity Coefficient (DSC) and Relative Recall (RR) and the lowest value for Average Surface Distance (ASD).

Postprocessing with the 3D level set improved the performance of each method. Specifically, 3D FCNNs + 3D level set showed a 16.44% improvement in DSC, 15.23% in RR, and a reduction in ASD from 4.2613 mm to 3.0029 mm. Similarly, 3D U-net + 3D level set and DenseVoxNet + 3D level set achieved improvements in DSC and RR, as well as significant reductions in ASD.

### Conclusion/Summary

This research introduces a novel 3D fully convolutional network, 3D MSDenseNet, for precise colorectal tumor segmentation in T2-weighted MRI volumes. The network employs dense interconnectivity between horizontal and vertical layers, capturing features of different resolutions throughout the network. Experimental results demonstrate superior performance compared to traditional CNNs, 3D U-net, and DenseVoxNet. Additionally, a 3D level-set algorithm is incorporated as a postprocessor, enhancing the segmentation results for all deep learning-based approaches. The proposed method, with a simple architecture totaling around 0.7 million parameters, outperforms DenseVoxNet and 3D U-net. Future work may involve validating the method on other medical volumetric segmentation tasks.

# Conclusion

In conclusion, the three methodologies discussed, Meta, RaBit, and 3D MSDenseNet, present unique contributions in polyp segmentation for the diagnosis of colorectal cancer. 

The Meta approach integrates transformer attention mechanisms into the UNet architecture, offering a blend of high accuracy and real-time applicability crucial for colonoscopy procedures. IT uses self-attention in local and global regions of the colon images to capture relations in both contexts. By addressing challenges in handling fine-detailed features and computational costs, Meta demonstrates exceptional segmentation accuracy across diverse datasets. However, its reliance on CNN backbone architecture may limit its effectiveness in capturing global context comprehensively.

In contrast, RaBit introduces a lightweight transformer model with bidirectional pyramid decoders, increasing polyp segmentation accuracy across multiple datasets. The incorporation of a reverse attention mechanism allows RaBit to selectively focus on relevant features while filtering out noise, leading to more accurate and robust feature representations. This mechanism plays a crucial role in capturing intricate details and boundaries of polyp structures in medical imaging data. Additionally, RaBit's bidirectional pyramid decoders enable the model to aggregate multi-scale contextual information, combining both global context and fine-grained details. Yet, the intricate network architecture of RaBit may entail higher computational demands and training complexities.

Meanwhile, 3D MSDenseNet captures both local and global contextual information with its 3D convolutional architecture. It used two parallel paths of 3D layers used to define features related to the depth and scale of the images. 3D convolutional layers improve colon segmentation in CT scans by considering spatial context across three dimensions, extracting features from the entire volume for comprehensive understanding, and integrating contextual information along the depth to reduce information loss and enhance segmentation accuracy. The integration of a 3D level-set post-processing algorithm further refines segmentation accuracy, making it a valuable tool for automated tumor segmentation. Nonetheless, the requirement for volumetric MRI data may constrain its applicability in certain clinical contexts compared to 2D approaches. 

In evaluating performance, each approach demonstrates significant advancements over conventional methods. The choice of the optimal approach hinges upon various factors, including segmentation accuracy, computational efficiency, and real-time feasibility. Meta excels in striking a balance between accuracy and efficiency, while RaBit showcases novel attention mechanisms, and 3D MSDenseNet underscores robustness in handling volumetric MRI data. Looking at the results from each of the models, Meta and RaBit can be compared as they were both ran on the Kvasir-SEG and ClinicDB dataset for binary polyp identification from the backgroun colon. RaBit performed with a DICE coefficient of 0.951 and an IOU of 0.911 on the ClinicDB dataset, and a DICE of 0.921 and IOU of 0.873 on the Kvasir-SEG dataset. On the other hand, Meta achieves a DICE of 0.939 and IOU of 0.890 on the ClinicDB dataset, and a DICE of 0.919 and IOU of 0.864 on the Kvasir-SEG dataset. 3D MSDenseNet achieves a DICE score of 0.841 although on a separate dataset. Overall these scores indicate that all three models perform well in accurately separating polyps within colons, with the transformer models of Meta and RaBit being slightly better than the 3D convolutional network of 3D MSDenseNet. The transformer-based models may produce better results due to their ability to capture long-range dependencies and contextual information across locally and globally, leading to more nuanced and accurate polyp identification and background colon differentiation compared to the 3D convolutional network.

Collectively, these methodologies hold great promise in the realm of colorectal cancer diagnosis. By automating and enhancing the accuracy of polyp segmentation, they contribute to earlier detection and treatment, thereby improving patient outcomes and reducing mortality rates. Furthermore, their development advances medical imaging technology, augmenting clinicians' capabilities in diagnosing and managing colorectal cancer effectively.

# References

[1] H. Wu, Z. Zhao and Z. Wang, "META-Unet: Multi-Scale Efficient Transformer Attention Unet for Fast and High-Accuracy Polyp Segmentation," in *IEEE Transactions on Automation Science and Engineering*, doi: 10.1109/TASE.2023.3292373. keywords: {Transformers;Image segmentation;Shape;Context modeling;Feature extraction;Task analysis;Colonoscopy;Polyp segmentation;META-Unet;efficient transformer;multi-scale transformer attention},

[2] Thuan, Nguyen Hoang, et al. “RaBiT: An Efficient Transformer Using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation.” *ArXiv (Cornell University)*, 12 July 2023, https://doi.org/10.48550/arxiv.2307.06420. Accessed 23 Mar. 2024.

[3] Mumtaz Hussain Soomro, Matteo Coppotelli, Silvia Conforto, Maurizio Schmid, Gaetano Giunta, Lorenzo Del Secco, Emanuele Neri, Damiano Caruso, Marco Rengo, Andrea Laghi, "Automated Segmentation of Colorectal Tumor in 3D MRI Using 3D Multiscale Densely Connected Convolutional Neural Network", *Journal of Healthcare Engineering*, vol. 2019, Article ID 1075434, 11 pages, 2019. https://doi.org/10.1155/2019/1075434

[4] Akilandeswari A, Sungeetha D, Joseph C, Thaiyalnayaki K, Baskaran K, Jothi Ramalingam R, Al-Lohedan H, Al-Dhayan DM, Karnan M, Meansbo Hadish K. Automatic Detection and Segmentation of Colorectal Cancer with Deep Residual Convolutional Neural Network. Evid Based Complement Alternat Med. 2022 Mar 17;2022:3415603. doi: 10.1155/2022/3415603. PMID: 35341149; PMCID: PMC8947925.