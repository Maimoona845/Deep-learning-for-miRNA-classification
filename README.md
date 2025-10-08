# miRNA Classification using Deep Learning

A convolutional neural network (CNN) implementation for classifying miRNA sequences by their target gene families using one-hot encoding and Keras with TensorFlow backend.

## Overview

This project implements a deep learning approach to classify microRNA (miRNA) sequences based on their target gene families. The system uses one-hot encoding to represent miRNA sequences and a CNN architecture for classification.

## Features

- **One-hot encoding** of miRNA sequences (A, U, G, C)
- **CNN architecture** with multiple convolutional and dense layers
- **Comprehensive evaluation** including accuracy metrics and confusion matrix
- **Model persistence** for saving and loading trained models
- **Visualization** of training history and results

## Model Architecture

The CNN model consists of:
- 3 Conv1D layers with batch normalization and max pooling
- Global max pooling layer
- 2 Dense layers with dropout for regularization
- Softmax output layer for multi-class classification

## Requirements

```bash
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
