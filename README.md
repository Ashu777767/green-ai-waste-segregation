# ♻️ Green AI–Based Smart Waste Segregation System

This project is a Green AI based application that uses Computer Vision and Deep Learning to classify waste and provide smart recycling guidance.

The system helps in identifying whether a waste item is **Biodegradable** or **Non-Biodegradable** and suggests appropriate recycling or disposal actions.

---

## 🔍 Problem Statement

Improper waste segregation leads to environmental pollution and inefficient recycling.
Manual waste sorting is slow, error-prone, and not scalable.

This project aims to provide an **AI-powered solution** for smart waste segregation.

---

## 💡 Solution Overview

- Uses a **Convolutional Neural Network (CNN)** with **MobileNetV2** for image classification
- Classifies waste into:
  - Biodegradable
  - Non-Biodegradable
- Applies a **Smart Green Logic Layer** to suggest:
  - Possible waste type
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
- Input size: 224 × 224
- Output: Binary classification (Biodegradable / Non-Biodegradable)

---

## 🌱 Green AI Features

- Reduces manual waste sorting
- Encourages proper recycling
- Provides sustainability-aware decisions
- Lightweight and deployable on edge devices

---

## 🖥️ Application Features

- Upload waste image and get prediction
- Real-time webcam waste classification (local)
- Confidence score for predictions
- Smart recycling recommendations

---

## 🚀 How to Run the Project Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/green-ai-waste-segregation.git
cd green-ai-waste-segregation
