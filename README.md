# SVM-Based Image Classification with Feature Descriptors

This project leverages Support Vector Machines (SVM) and feature descriptors to improve image classification accuracy across four categories: animals, nature, people, and man-made objects. By integrating Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), and Color Histograms, we address high computational costs and generalization limitations of traditional deep learning models.

## Project Overview

Using the *MIT-Adobe 5K* dataset, images were preprocessed, labeled, and split into training (80%) and testing (20%) sets, with standardized dimensions of 128x128. The SVM classifier, initially achieving 51% accuracy, improved to 55% after augmenting the animals category. Feature descriptors capture essential edge, shape, and color information, allowing SVM to classify images efficiently.

## Methodology

1. *Feature Extraction*: HOG, SIFT, and Color Histograms are applied to capture detailed edge, shape, and color distribution features.
2. *Classification*: SVM processes these features for robust classification with low computational demand.

### Comparative Analysis

Additional models were evaluated for comparison:

| Model            | Accuracy |
|------------------|----------|
| SVM              | 55%      |
| CNN              | 85%      |
| K-Nearest Neighbors (KNN) | 41% |
| RBF Kernel SVM   | 67%      |
| Random Forest    | 63%      |
| Decision Tree    | 42%      |

## Results and Conclusion

While the SVM model improved to 55% accuracy after data augmentation, CNN outperformed with an 85% accuracy. This project highlights the practicality of feature-based SVM models as efficient alternatives to deep learning models in specific image classification tasks.

