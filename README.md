# CLAD: Confidence-based self-Labeling Anomaly Detection


This is a [PyTorch](http://pytorch.org) implementation for [\<What is Wrong with One-Class Anomaly Detection?\>](https://arxiv.org/abs/2104.09793) published at ICLR 2021 Workship on Security and Safety in Machine Learning Systems.

## Citation
```
```

## Abstract

> > From a safety perspective, a machine learning method embedded in real-world applications is 
> > required to distinguish irregular situations. For this reason, there has been a growing 
> > interest in the anomaly detection (AD) task. Since we cabbit observe abnormal samples for 
> > most of the cases, recent AD methods attemp to formulate it as a task of classifying whether 
> > the sample is normal or not. However, they potentially fail when the given normal samples 
> > are inherited from diverse semantic labels. To tackle this problem, we introduce a latent 
> > class-condition-based AD scenario. In addition, we propose a confidence-based self-labeling 
> > AD framework tailored to our proposed scenario. Since our method leverages the hidden class 
> > information, it successfully avoids generating the undesirable loose decision region that 
> > one-class methods suffer. Our proposed framework outperforms the recent one-class AD methods 
> > in the latent multi-class scenarios.


### Requirements

* Python 3.8
* PyTorch 1.7.1

To install all the required elements, run the code:
```
pip install -r requirements.txt
```

### Run scripts

Implemented dataset list : MNIST, GTSRB, CIFAR-10, Tiny-ImageNet

To run the experiments, run the scripts:
```
cd <path-to-CLAD-directory>

# change to run_scripts directory
cd src/run_scripts

# run MNIST experiments
sh run_mnist.sh

# run GTSRB experiments
sh run_gtsrb.sh

# run CIFAR-10 experiments
sh run_cifar10.sh

# run Tiny-ImageNet experiments
sh run_tiny_imagenet.sh
```



## Scenario (Latent Class-condition Anomaly Detection Scenraio)

To reduce the gap between the real-world and the one-class AD senarios, we simulate the sceanrio environment where the latent sub-classes exist implicitly. With this environment, it is crucial to learn a decision boundary by seeing not only the normality of the data samples but also its semantics. Note that such class information is not observable, thus the AD framework may require learning the semantic representation in an unsupervised or self-supervised manner.

<p align="center">
<img src="./figures/fig_scenario_overview.png" width="1000">
</p>
          


## CLAD (Confidence-based self-Labeling Anoamly Detection)

We propose a Confiedence-based self-Labeling Anomaly Detection (CLAD) framework with four states illustrated in the figure below.

<p align="center">
<img src="./figures/fig_model_overview.png" width="1000">
</p>



## Categorizing Each Dataset

We devised super-categories by merging the semantic labels to simulate our AD scenario as illustrated in the figure and table below. 

<p align="center">
<img src="./figures/fig_dataset_scenario.png" width="1000">
</p>

<p align="center">
<img src="./figures/table_scenario.png" width="1000">
</p>



## Experimental Results

We compare with one-class AD methods: [OCSVM](https://direct.mit.edu/neco/article/13/7/1443/6529/Estimating-the-Support-of-a-High-Dimensional), [OCNN](https://link.springer.com/chapter/10.1007/978-3-319-71249-9_3), [OCCNN](https://arxiv.org/abs/1901.08688), [SVDD](https://dl.acm.org/doi/10.1023/B%3AMACH.0000008084.60811.49), and [DeepSVDD](http://data.bit.uni-bonn.de/publications/ICML2018.pdf).

### Proposed Scenario

<p align="center">
<img src="./figures/table_scenario.png" width="1000">
</p>

### One-Class Classification

<p align="center">
<img src="./figures/table_ablation_one_class.png" width="1000">
</p>

### ablation study on hyper-parameters

<p align="center">
<img src="./figures/fig_ablation_hyperparameter.png" width="1000">
</p>

### Implementation Details

For latent feature extraction, we used convolutional autoencoder architecture.
For clustering visualization, we used [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE) for efficiency issue.
For self-labeling via clustering, we mimicked the approach of the [DEC](https://github.com/piiswrong/dec) to self-assign labels to data samples.
For classifier for confidence-based AD, we used [ResNet-18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).
For the scaling the confidence scores, we adopted the OOD mechanism from the [ODIN](https://github.com/facebookresearch/odin) to gain more robust AD score.
* Note that the hyper-parameters may vary depending on the scenarios for each dataset.
