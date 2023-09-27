# Sign Language MNIST Classification

This project aims to classify hand gestures in sign language using various deep learning models. The dataset used is the Sign Language MNIST dataset, which consists of grayscale images of hand signs representing different alphabets (A-Z).

## Code Overview

### Convolutional Neural Network (CNN) Model

- The `CNN_model` function defines a CNN architecture for image classification.
- It consists of convolutional layers followed by max-pooling and dropout layers to prevent overfitting.
- The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss for multi-class classification.
- The model is trained using the training dataset and validated using the validation dataset.
- Training history is stored in `history` for loss and accuracy evolution.

### Fully Connected Neural Network Model

- The `fc_model` Sequential model is defined for image classification using fully connected layers.
- It starts with flattening the input and adding fully connected layers with dropout.
- The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.
- Similar to the CNN model, the training and validation process is performed and history is stored in `history_fc`.

### Model Evaluation

- Model accuracy is evaluated on the training, test, and validation datasets.
- The `accuracy_score` function from sklearn.metrics is used to calculate the accuracy of the predictions.

### Visualization

- Matplotlib is used to visualize the loss and accuracy evolution during training.
- Plots show the trend of loss and accuracy on both the training and validation datasets.

### Results

The models are trained and evaluated on the Sign Language MNIST dataset. The achieved accuracies on different datasets are as follows:

- **CNN Model**:
  - Training Accuracy: 100%
  - Test Accuracy: 95.93%
  - Validation Accuracy: 100%

- **Fully Connected Model**:
  - Training Accuracy: 96.99%
  - Test Accuracy: 71.36%
  - Validation Accuracy: 96.74%

## Conclusion

This code showcases the implementation of CNN and fully connected neural network models for sign language classification. The CNN model demonstrates higher accuracy on test and validation datasets compared to the fully connected model. The provided code serves as a starting point for further experimentation and improvement.
