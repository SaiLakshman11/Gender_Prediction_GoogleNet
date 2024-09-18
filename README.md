# Gender_Prediction_GoogleNet

Summary: Developed a gender prediction system with Deep learning architecture -- GoogleNet.

Problem Statement: A dataset containing over 23,000 images of individuals aged between 1 and 116 years, and the goal is to develop a model to predict the gender of individuals from these images. The dataset is organized such that each image file is named with an age, gender, and other details. The genders are represented as 0 (male) and 1 (female). The challenge is to create a gender prediction model using Convolutional Neural Networks (CNNs) and evaluate its performance.

Steps Involved:

1. Data Preparation:
   Load Dataset: The dataset is loaded from a directory containing images. The total number of files is counted, and the files are shuffled and split into training and testing sets.
   Create Temporary Directories:
   Temporary directories are created to organize the images into folders based on gender labels (0 for male and 1 for female). This organization helps in feeding the data into the Keras ImageDataGenerator.

2. Data Augmentation and Generator:
   ImageDataGenerator: An ImageDataGenerator is used to rescale pixel values to the range [0, 1] and to create data generators for training and testing datasets. The training data generator performs image augmentation, which helps improve model generalization.
   Generate Training Data: Training data is loaded using train_generator, and its shape is printed to confirm the data loading.

3. Building Model:
   GoogleNet Architecture: A CNN model based on the GoogleNet architecture is built using Keras. This model includes several convolutional layers, max-pooling layers, batch normalization, dropout layers, inception layers and fully connected dense layers.

4. Model Training: Fit the Model: The model is trained using the training data generator for 20 epochs. The training process includes monitoring the model's performance on the training set.

5. Test Data Preparation:
   Prepare Test Data: Similar to training data, the test images are organized into appropriate subdirectories. A test data generator is created to load and preprocess the test images.

6. Model Evaluation:
   Evaluate Model: The model's performance is evaluated on the test set using accuracy. Predictions are made for test images, and the model's performance metrics such as precision, recall, F1 score, true positive rate, and false positive rate are computed.
   Confusion Matrix: A confusion matrix is plotted to visualize the model's classification performance.

7. Prediction on Specific Images:
   Several images are loaded and preprocessed. The trained model predicts the gender of these images, and the results are compared with the original labels. The images and predictions are displayed.

Thank you for reading!!!
