# Raspberry Pi 4 Clothing Classification System

## Overview

This project implements a real-time clothing detection and classification system using a Raspberry Pi 4 and Raspberry Pi Camera. The system captures images and determines whether the object is a clothing item using a custom-built Convolutional Neural Network (CNN).

## Features

* Real-time image capture using Raspberry Pi Camera
* Image-based clothing classification
* Custom CNN implementation (no heavy frameworks like TensorFlow)
* Lightweight design suitable for embedded systems

## Technologies Used

* Python
* Raspberry Pi 4 (64-bit OS)
* Raspberry Pi Camera Module
* NumPy (for numerical operations)

## Project Structure

* `classify_camera.py` – Captures live images and performs classification
* `classify_image.py` – Classifies static images
* `train_model.py` – Trains the CNN model
* `cnn_mac.py` – CNN architecture and forward propagation logic

## How It Works

1. Images are captured using the Raspberry Pi Camera or loaded from files.
2. Images are preprocessed (resized, normalized).
3. The CNN extracts features from the image.
4. The final layer predicts whether the object is clothing or not.

## Setup Instructions

### 1. Raspberry Pi Setup

* Install Raspberry Pi OS (64-bit recommended)
* Enable camera:

```bash
sudo raspi-config
```

→ Interface Options → Enable Camera

### 2. Install Dependencies

```bash
pip install numpy opencv-python
```

### 3. Train the Model

```bash
python src/train_model.py
```

### 4. Run Image Classification

```bash
python src/classify_image.py
```

### 5. Run Real-Time Classification

```bash
python src/classify_camera.py
```

## Dataset

The model can be trained using:

* Custom dataset (recommended for better performance)
* Fashion-MNIST dataset (for initial testing)

## Results

* Successfully detects clothing vs non-clothing items
* Designed for real-time performance on Raspberry Pi 4
* Lightweight model suitable for low-power devices

## Challenges

* Limited processing power on Raspberry Pi
* Building a CNN without high-level libraries
* Real-time image processing constraints

## Future Improvements

* Expand dataset for better accuracy
* Multi-class classification (e.g., shirt, pants, shoes)
* Optimize model for faster inference
* Add bounding box detection for object localization

## Author

Malcolm Howard
University of Central Arkansas
