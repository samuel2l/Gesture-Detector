# Gesture Detector - Unlock the Power of Gesture Recognition

Welcome to the **Gesture Detector** project, where machine learning meets real-time interaction! This repository is designed to help you dive into the exciting world of computer vision and gesture recognition. Whether you're building a gesture-controlled application or simply exploring the capabilities of OpenCV, this project is a perfect starting point.

The **Gesture Detector** uses your hand movements or any custom gestures to perform actions in real time. By training your system to recognize specific gestures, you can create intuitive interfaces or control systems using nothing but your hands (or other body parts)! With just a camera and a few lines of code, you’ll be able to control devices or trigger actions using gestures.


## Introduction

The **Gesture Detector** project is built with Python and OpenCV, providing an easy-to-understand framework for training and testing gesture recognition systems. Whether you're a beginner in computer vision or an expert looking for an easy-to-use prototype, this project is the perfect tool to explore the possibilities of gesture-based interactions.

### Key Features:
- **Custom Gesture Dataset**: Capture and define your own gestures using the `dataset.py` script.
- **Real-Time Inference**: Use `inference.py` to recognize gestures from your camera feed and trigger actions or control systems.
- **Expandability**: Easily adapt the code to recognize different gestures or integrate with other applications.

## How It Works
To run this project you will need to change the paths in the train,inference and dataset.py files to point to actual locations on your laptop.
Run the dataset.py to train on your signals
Run the train.py to train the model on your signals
Run the inference.py to test your model in realtime.(Change path of model too in the inference.py depending on where it was saved)

At its core, the **Gesture Detector** consists of two primary components:
1. **Dataset Management** (`dataset.py`): This script allows you to create and manage a custom set of gestures. You can define new gestures, capture images, and store them in an organized manner.
2. **Inference** (`inference.py`): Once you've captured your custom gestures, this script performs real-time recognition using the trained model and attempts to classify new gestures as they are detected via the camera.

The process starts with capturing images of different gestures, followed by training a model to recognize them. Afterward, you can use the `inference.py` script to detect these gestures live and perform real-time recognition.
