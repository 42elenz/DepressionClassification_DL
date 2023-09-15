# DepressionClassification_DL

## Introduction
In this project, I aimed to predict Major Depressive Disorder (MDD) based on an open dataset of electroencephalograms (EEGs). I not only developed a basic Convolutional Neural Network (CNN) model for classification but also explored advanced techniques, including swarm learning with HPE (Hewlett Packard Enterprise) and generating new EEGs using a Diffusion model.

## Chapter 1: Data Preprocessing
In this chapter, I performed essential data preprocessing tasks to prepare the EEG data for analysis. The preprocessing involved:
- Tuning windowing parameters to achieve optimal segmentation of EEG signals for upsampling.

<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/hyperparameter" width=50%>

-  Removing unnecessary channels.
-  Applying differnt filters (a notch filter, high and lowband filter, artifact removal with mne-library).
-  Filter for channels that could be potentially labeled wrongly with trusted learning with cleanlab

<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/cleanlab.png" width=50%>

## Chapter 2: Basic CNN Model
In this chapter, I created a basic CNN model for MDD classification using the preprocessed EEG data. The model achieved an accuracy of up to 91 percent when coupled with the right preprocessing techniques.

<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/CNN.png" width=50%>

## Chapter 3: Swarm Learning with HPE
This chapter focuses on the distribution of data among peers in a swarm learning solution provided by HPE. Unfortunately, due to confidentiality agreements from my internship at HPE, I can't share specific implementation details or outputs. However, I can confirm that the results were comparable to those achieved by the local algorithm.
<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/Aufbau1.png" width=50%>

<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/Aufbau2.png" width=50%>

## Chapter 4: Diffusion Model for EEG Generation
In this chapter, I developed my own diffusion model tailored to work with 2D EEG data. The model performed well, successfully reproducing EEG signals with some notable observations:
- Specific noising step for EEG data
- 
<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/diffusionprocess.png" width=50%>

- Overfitting on a single EEG file yields the same EEG with a bit of noise which proves the functionallty of my U-Net
- Some runs which specific EEGs generated EEGs with similar frequency bands but higher noise levels.

<img src="https://github.com/42elenz/DepressionClassification_DL/tree/master/Diffusion.png" width=50%>

- An attempt to use a 1D Unet model yielded mixed channel information, which requires further investigation.

Please note that certain details and results may be limited or confidential due to the nature of the project.
