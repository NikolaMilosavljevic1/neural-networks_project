# Card classification using convolutional neural networks
The goal of this project was to classify card images into 53 classes using Convolutional Neural Networks (CNNs).  
The dataset used for this project can be found on Kaggle: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
<br><br>
The text of the project can be seen in *assignment.pdf*<br>
More details about model architecture, training configuration, results and performances are available in *report.pdf*
## Preprocessing
- All images are resized to 128x128x3
- Dataset is split into **train (60%), validation (20%) and test (20%)**
- Data augmentation applied (rotation, zoom, translation, horizontal flip)
## Training Configuration
- Loss Function: **SparseCategoricalCrossentropy()**
- Optimizer: **Adam**
- Regularization: **L2**
## Overfitting Prevention
- **Dropout** Layers were used after convolution blocks
- **L2 regularization** was applied to dense layers
- **Early stopping** was implemented
