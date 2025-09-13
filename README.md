# PneumoniaDetectionCNN: CNN-Based Pneumonia Detector

PneumoniaDetectionCNN is a deep learning project that classifies chest X-Ray images into two categories: **Pneumonia** and **Normal**.  
It demonstrates the use of Convolutional Neural Networks (CNNs) in medical imaging for accurate and accessible diagnostic assistance.

---

## Demo

[View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-computervision-cnn-activity-7372583616125730816-Vc0M?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)

---

## Project Overview

The workflow covers a complete CNN-based deep learning pipeline:

- **Dataset Cleaning**: Re-splitting for balanced training/validation/testing  
- **Deduplication**: Detection and removal of duplicate X-ray images  
- **Exploration & Visualization**: Class distribution and image inspection  
- **Data Augmentation**: Rotation, shift, zoom, and flipping  
- **Model Training**: CNN with multiple Conv2D layers, EarlyStopping, and ModelCheckpoint  
- **Deployment**: Interactive Streamlit app and Hugging Face Spaces for real-time prediction

### Performance Metrics
- **Accuracy**: `0.9606`  
- **Precision**: `0.9832`  
- **Recall**: `0.9624`  
- **F1 Score**: `0.9727`

---

## Project Workflow

- **Exploration & Visualization**: Class distribution analysis, duplicate image removal  
- **Preprocessing**: Image resizing `(224×224)`, grayscale conversion, rescaling `(1/255)`  
- **Augmentation**: Rotation, width/height shift, zoom, and horizontal flipping  
- **Modeling**: CNN with Conv2D + ReLU, MaxPooling, Flatten, and Dense layers  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score  
- **Deployment**: Streamlit app & Hugging Face Spaces for predictions

---

## Dataset

- **Source**: [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  
- **Classes**: Pneumonia, Normal  
- **Image Size**: Resized to `(224 × 224)`, grayscale  
- **Preprocessing**: Dataset re-split, duplicates removed, visual exploration performed  

---

## Real-World Impact

This project demonstrates the application of CNNs in healthcare imaging, particularly for Pneumonia detection.  
By enabling early detection, it supports timely treatment and better patient outcomes.

---

## Project Links

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/pneumoniadetectioncnn-chest-x-ray-classifier)  
- **Streamlit App**: [Try it Now](https://pneumoniadetectioncnn-qxz3ree3z3kjz5swg5gsmt.streamlit.app/)  
- **Hugging Face Repo**: [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/PneumoniaDetectionCNN)  

---

## App Preview

![Streamlit App](https://github.com/rawan-alwadiya/PneumoniaDetectionCNN/blob/main/App.png)  
![Pneumonia Detection Example](https://github.com/rawan-alwadiya/PneumoniaDetectionCNN/blob/main/PNEUMONIA.png)

---

## Tech Stack

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- TensorFlow/Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- CNN (Convolutional Neural Network)  
- Data Augmentation  
- Model Evaluation (Accuracy, Precision, Recall, F1 Score)  
- Streamlit & Hugging Face Deployment
