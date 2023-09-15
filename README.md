# DepressionClassification_DL

## Introduction
In this project, I aimed to predict Major Depressive Disorder (MDD) based on an open dataset of electroencephalograms (EEGs). I not only developed a basic Convolutional Neural Network (CNN) model for classification but also explored advanced techniques, including swarm learning with HPE (Hewlett Packard Enterprise) and generating new EEGs using a Diffusion model.

## Chapter 1: Data Preprocessing
In this chapter, I performed essential data preprocessing tasks to prepare the EEG data for analysis. The preprocessing involved:
- Tuning windowing techniques to achieve optimal segmentation of EEG signals (explained why windowing is necessary).
- Removing unnecessary channels.
- Applying a notch filter to eliminate noise.

## Chapter 2: Basic CNN Model
In this chapter, I created a basic CNN model for MDD classification using the preprocessed EEG data. The model achieved an accuracy of up to 91 percent when coupled with the right preprocessing techniques.

## Chapter 3: Swarm Learning with HPE
This chapter focuses on the distribution of data among peers in a swarm learning solution provided by HPE. Unfortunately, due to confidentiality agreements from my internship at HPE, I can't share specific implementation details or outputs. However, I can confirm that the results were comparable to those achieved by the local algorithm.

## Chapter 4: Diffusion Model for EEG Generation
In this chapter, I developed my own diffusion model tailored to work with 2D EEG data. The model performed well, successfully reproducing EEG signals with some notable observations:
- Overfitting on a single EEG file.
- Variable performance on runs with different numbers of input EEGs.
- Some runs generated EEGs with similar frequency bands but higher noise levels.
- An attempt to use a 1D Unet model yielded mixed channel information, which requires further investigation.

Please note that certain details and results may be limited or confidential due to the nature of the project.
