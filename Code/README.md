# [Code Structure](https://github.com/iSiddharth20/DeepLearning-ImageClassification-Toolkit)

### 1. PreProcessing.ipynb
- Load Dataset from 'DATA_DIR' Directory
- Creates DataFrame containing Full Paths of Images and their Class Labels
- (Change as per Requirement) Rescale Images to Computationally Efficient Resolution
- (Optional but Recommended) Extracts Largest Object from Image using 'image_processing' Function
    - Leverages Parallel Processing for Faster Results
- Compares Original and Rescaled+Processed Image SIde-By-Side to make necessary changes
- Converts Processed Images to NumPy Array and Exports as Pickle File
    - Verifies If Exported Pickle File is Appropriate through 10 Random Samples
- (Optional) Merge Certain Class Lables Together
- Split Data for Training, Testing, Validation with Stratify to ensure data balancing
    - Verify if Split is Appropriate through 2 random samples
- (Optional) Perform Random Oversampling on Data to reduce Biasness
    - Verify if Oversampling is Appropriate through 2 random samples
- Perform One-Hot-Encoding of Class Labels
- Training, Testing, Validation Data and One-Hot-Encoding is Exported as Pickle Files

### 2. Modeling_EfficientNetB0.ipynb
### 2. Modeling_InceptionResNetV2.ipynb
### 2. Modeling_ResNet50.ipynb
### 2. Modeling_VGG16.ipynb
- Training, Testing, Validation Data and One-Hot-Encoding are Imported
    - All Data is converted to TensorFlow Format
- (Change as per Requirement) Learning Rate Scheduler is Defined
- (Change as per Requirement) Stochastic Gradient Descent with Momentum is Used as Optimizer
- (Change as per Requirement) Added Data Augmentation Techniques to improve Model Learning
- Final Model is Created
    - Base Model is Loded from TensorFlow Library
    - Custom Optimal Changes have been made to the Structure
    - Tensorflow strategy.scope() is used to Distribute the Training of Model on All Available GPUs
    - Final Model is Compiled
- Final Model is Trained
    - Final Model with Lowest Validation Loss is Exported as a '.h5' file
- (Optional) Trained Model can be Retrained for Reinforcement Learning
- Traning Time (In Seconds) is Displayed
- (Optional) Save the Cell Output of Model Training in 'LOGS_DIR' in respective Model Name text file to generate graphs (discussed further in '4. Generate Graphs.ipynb')

### 3. Verification and Confusion Matrix.ipynb
- Trained Model and One-Hot-Encoding are Imported
- Entire Dataset is Run through the Trained Model to get Ground Truth of Accuracy
- (Optional) Incorrectly Classified Image Files will be copied to a seprate folder with detected class label
- Ground Truth Classification Confusion Matrix is Created
    - (Optional) Confusion Matrix can be Exported as a '.png' file

### 4. Generate Graphs.ipynb
- Import Training Logs from 'LOGS_DIR' Directory 
- Generate Graphs comparing performances of Models
    - Graphs are generated for Validation Loss, Validation Accuracy, Validation F-1 Score, Validation AUC
    - (Optional) Graphs can be exported as '.png' file

### HelperFunctions .py
- Function to display 2 images side-by-side on screen
- Function to Extract largest object from souruce image

### LearningRateScheduler .py (Extra)
- Choose whichever fits your requirement the best :
    - Manual 
    - ReduceLROnPlateau
    - Cosine Annealing with Warm Restarts (Best if Paired with SGD with Momentum)
    - Cosine Annealing