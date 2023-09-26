# DeepLearning-ImageClassification-Toolkit
End-to-end Image Classification using Deep Learning toolkit for custom image datasets. Features include Pre-Processing, Training with Multiple CNN Architectures and Statistical Inference Tools. Special utilities for RAM optimization, Learning Rate Scheduling, Detailed Code Comments and Necessary Diagrams are included.
This Codebase requires [TensorFlow](https://www.tensorflow.org/install), kindly make sure you have that installed.

### Overview of Complete System :
![Project Description](./Diagrams/Complete%20System%20Overview.png)

### PreProcessing Major Highlights:
- Remove Background and Extract Object from Source Image
- Convert Image into NumPy Array
- Split Data into Training, Testing and Validation Data
- Create One-Hot-Encoding for Categorical Labels
- Provided Functionalities for Oversampling
- Provided Comments for Better Understanding
- More Features and Detailed Explanation Available in Code

### Models Trained:
- EfficientNetB0
- InceptionResNetV2
- ResNet50
- VGG16

### Statistical Analysis for:
- Generate Graphs for Validation Loss, Validation Accuracy, F-1 Score, Validation AUC
- Generate Confusion Matrix for Each Model