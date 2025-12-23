# Handwritten Digit Classification with Deep Learning (MNIST)

This project demonstrates how to build and train a deep neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Objective
The goal is to design a feedforward neural network capable of accurately recognizing digits (0–9) from grayscale images and evaluate its performance using standard metrics.

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The MNIST dataset contains:
- 60,000 training images
- 10,000 test images  
Each image is 28x28 pixels representing handwritten digits from 0 to 9.

## Model Architecture
The neural network is composed of:
- Input layer (28x28 images)
- Flatten layer
- Dense layer with 128 neurons (ReLU)
- Dense layer with 64 neurons (ReLU)
- Output layer with 10 neurons (Softmax)

Optimizer: Adam  
Loss function: Sparse Categorical Crossentropy

## Training
- Epochs: 5  
- Batch size: 128  
- Validation split: 20%

Training and validation curves are plotted to monitor loss and accuracy over epochs.

## Results
- Test Accuracy: **~97.3%**
- The model shows strong generalization performance on unseen data.
- Confusion matrix is used to analyze prediction errors per class.

## Files
- `mnist_deep_learning.ipynb` — Main Jupyter Notebook with data preparation, model training and evaluation.

## Notes
This project was developed as part of a Deep Learning learning path, focusing on understanding neural network implementation and evaluation using TensorFlow and Keras.
