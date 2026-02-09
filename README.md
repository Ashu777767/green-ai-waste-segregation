# ♻️ Green AI–Based Smart Waste Segregation System

This project is a **Green AI–based application** that uses **Computer Vision** and **Deep Learning** to classify waste and provide smart recycling guidance.

The system identifies whether a waste item is **Biodegradable** or **Non-Biodegradable** and suggests appropriate recycling or disposal actions.

---

## 🔍 Problem Statement

Improper waste segregation leads to environmental pollution and inefficient recycling.  
Manual waste sorting is slow, error-prone, and not scalable.

This project aims to provide an **AI-powered solution** for smart waste segregation.

---

## 💡 Solution Overview

- Uses a **Convolutional Neural Network (CNN)** with **MobileNetV2**
- Waste classification:
  - Biodegradable
  - Non-Biodegradable
- Implements a **Smart Green Logic Layer** to suggest:
  - Waste type
  - Recycling possibility
  - Recommended disposal bin
- Supports:
  - Image upload prediction
  - Live camera classification (local deployment)

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- NumPy  
- Pillow  

---

## 📊 Model Details

- Model: CNN with MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224
- Output: Binary Classification  
  - Biodegradable  
  - Non-Biodegradable  

---

## 🌱 Green AI Features

- Reduces manual waste sorting
- Encourages proper recycling habits
- Lightweight and energy-efficient model
- Suitable for edge and local deployment

---

## 🖥️ Application Features

- Upload waste image for prediction
- Real-time webcam waste classification
- Confidence score for predictions
- Smart recycling recommendations

---

## 🚀 How to Run the Project Locally

1. Clone the repository:
```bash
git clone https://github.com/Ashu777767/green-ai-waste-segregation.git
cd green-ai-waste-segregation
