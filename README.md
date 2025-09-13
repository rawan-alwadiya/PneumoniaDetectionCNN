# **PneumoniaDetectionCNN: CNN-Based Pneumonia Detector**

PneumoniaDetectionCNN is a deep learning project that applies a **Convolutional Neural Network (CNN)** to classify chest X-Ray images into two categories: **Pneumonia** and **Normal**.  
The project demonstrates an **end-to-end computer vision workflow** including **dataset cleaning, augmentation, CNN modeling, evaluation, and deployment with Streamlit & Hugging Face**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-computervision-cnn-activity-7372583616125730816-Vc0M?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://pneumoniadetectioncnn-qxz3ree3z3kjz5swg5gsmt.streamlit.app/)  
- 🤗 [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/PneumoniaDetectionCNN)  

![App Demo](https://github.com/rawan-alwadiya/PneumoniaDetectionCNN/blob/main/App.png)  
![Pneumonia Detection Example](https://github.com/rawan-alwadiya/PneumoniaDetectionCNN/blob/main/PNEUMONIA.png)

---

## **Project Overview**

The workflow includes:  
- **Dataset Cleaning**: re-splitting for balanced train/val/test sets  
- **Deduplication**: detection and removal of duplicate X-ray images  
- **Exploration & Visualization**: class distribution and image inspection  
- **Data Augmentation**: rotation, shift, zoom, horizontal flipping  
- **Modeling (CNN)**: Conv2D + ReLU, MaxPooling, Flatten, Dense layers  
- **Evaluation**: accuracy, precision, recall, F1-score  
- **Deployment**: interactive **Streamlit web app** and Hugging Face Spaces for real-time predictions  

---

## **Objective**

Develop and deploy a robust **CNN-based classification model** to detect Pneumonia in chest X-ray scans, ensuring **high accuracy** and **clinical relevance**.

---

## **Dataset**

- **Source**: [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)  
- **Classes**: Pneumonia, Normal  
- **Image Size**: resized to `(224 × 224)`, grayscale  
- **Preprocessing**: dataset re-split, duplicates removed, visual exploration performed  

---

## **Project Workflow**

- **EDA & Visualization**: class distribution, duplicate removal, sample image inspection  
- **Preprocessing**:  
  - Image resizing `(224×224)`  
  - Grayscale conversion  
  - Rescaling `(1/255)`  
- **Augmentation**: random rotation, width/height shift, zoom, horizontal flip  
- **Modeling (CNN)**:  
  - Conv2D + ReLU with increasing filters `(64 → 128 → 256 → 512)`  
  - MaxPooling2D after convolutional blocks  
  - Flatten → Dense `(256 → 64 → 1)` with **Sigmoid output**  
- **Training Setup**:  
  - Optimizer: Adamax `(lr=0.001)`  
  - Loss: Binary Crossentropy  
  - Callbacks: EarlyStopping (patience=10, restore best weights) and ModelCheckpoint (`best_model.h5`)  
  - Epochs: 30  

---

## **Performance Results**

**Convolutional Neural Network Classifier:**  
- **Accuracy**: `0.9606`  
- **Precision**: `0.9832`  
- **Recall**: `0.9624`  
- **F1-score**: `0.9727`  

The model achieved **high precision and recall**, ensuring reliable Pneumonia detection while minimizing false positives.

---

## **Project Links**

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/pneumoniadetectioncnn-chest-x-ray-classifier)  
- **Streamlit App**: [Try it Now](https://pneumoniadetectioncnn-qxz3ree3z3kjz5swg5gsmt.streamlit.app/)  
- **Hugging Face Repo**: [Explore on Hugging Face](https://huggingface.co/RawanAlwadeya/PneumoniaDetectionCNN)  

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- TensorFlow / Keras, scikit-learn  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- CNN (deep convolutional neural network with multiple Conv2D blocks)  
- Data Augmentation (rotation, shift, zoom, flipping)  
- EarlyStopping & ModelCheckpoint  
- Real-time deployment with Streamlit & Hugging Face  
