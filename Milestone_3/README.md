README – Milestone 3
TraceFinder: CNN-based Scanner Identification (Training and Evaluation)
Objective

The goal of Milestone 3 is to design, train, evaluate, and explain a deep learning model that can identify the source scanner of a document image based on forensic artifacts.

Key Contributions

Designed a custom convolutional neural network for grayscale document images

Used image-wise dataset splitting to prevent data leakage

Achieved high validation and test accuracy

Performed image-wise voting-based evaluation

Implemented Grad-CAM for model explainability

Dataset Structure
dataset/
├── Cannon120-1/
├── Cannon200/
├── EpsonV500/
├── Epsonv39-1/
└── Hp/


Each folder represents one scanner class

Images are raw scanned document images

Dataset split: 70% training, 15% validation, 15% testing

Splitting is done at image level, not patch level

Model Architecture

Input size: 224 × 224 grayscale images

Four convolutional blocks

Convolution, Batch Normalization, ReLU

Second convolution, Batch Normalization, ReLU

Max pooling and dropout

Global Average Pooling layer

Fully connected classifier (256 → 128 → number of classes)

Total parameters: approximately 1.2 million

Training Configuration

Optimizer: Adam

Loss function: Weighted Cross Entropy Loss

Learning rate scheduler: ReduceLROnPlateau

Early stopping enabled

Training performed on CPU or GPU depending on availability

Evaluation Strategy

Accuracy and loss curves for training and validation

Image-wise evaluation using majority voting

Confusion matrix visualization

Per-class precision, recall, and F1-score

Generated output files:

best_forensic_cnn.pth
training_curves.png
confusion_matrix.png

Model Explainability (Grad-CAM)

Grad-CAM applied to the final convolutional layer

Highlights scanner-specific artifact regions

Provides forensic interpretability for predictions

Generated output:

gradcam_output.png

Milestone 3 Outcome

Successfully trained a robust CNN model

Achieved strong generalization performance

Added explainability suitable for forensic analysis

Model ready for deployment in Milestone 4
