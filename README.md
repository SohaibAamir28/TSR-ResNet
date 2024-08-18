# TSR-ResNet
Traffic Sign Recognition Using Deep Learning ResNet
# Traffic Sign Classification Using ResNet

![Traffic Sign Classification Example](https://github.com/SohaibAamir28/TSR-ResNet/output/model.png)

This project demonstrates how to build a traffic sign classification system using a Convolutional Neural Network (CNN) with the ResNet architecture. The model is trained on traffic sign images and is designed to operate accurately under various environmental conditions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Traffic sign recognition is a critical component of autonomous driving systems. This project leverages deep learning techniques, specifically the ResNet architecture, to classify traffic signs accurately in real-time.

## Dataset

The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The dataset consists of images of traffic signs collected under various conditions.

- **Data Augmentation**: Applied techniques like rotation, scaling, and brightness adjustment to enhance model robustness.

## Model Architecture

The project uses the ResNet50 architecture, pre-trained on the ImageNet dataset. The model is fine-tuned on the traffic sign dataset to improve accuracy.

- **Layers**: Global Average Pooling, Dense (ReLU), Dense (Softmax for classification).
- **Optimization Techniques**: Batch normalization, dropout, transfer learning.

## Training

The model is trained using the following parameters:
- **Learning Rate**: `0.001`
- **Batch Size**: `32`
- **Epochs**: `10` (additional fine-tuning for 5 epochs)

Model training includes data augmentation and hyperparameter optimization to improve performance.

## Evaluation

The model is evaluated on a validation set with metrics such as accuracy, precision, recall, and F1-score. The final model is suitable for deployment on embedded systems like NVIDIA Jetson or Raspberry Pi.

## Dependencies

To install the required dependencies, create a virtual environment and run:

```bash
pip install -r requirements.txt
