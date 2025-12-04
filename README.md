# SkinSavant-AI-Powered-Dermatology-for-Early-Skin-Cancer-Detection
SkinSavant leverages deep learning (DenseNet169) on the HAM10000 dataset  to provide early detection capabilities for 7 types of skin lesions,  including melanoma and basal cell carcinoma.


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

### 🩺 Overview
SkinSavant is an AI-powered platform designed to assist in early 
detection of skin cancers by analyzing dermatoscopic images. Using 
a fine-tuned DenseNet169 model trained on the HAM10000 dataset, 
it classifies lesions into 7 categories with high accuracy.

### 📊 Performance
- **Dataset**: HAM10000 (10,015 images, 7 classes)
- **Model**: DenseNet169 with custom classifier head
- **Accuracy**: 90%+ on validation set
- **Key Detection**: Melanoma, Basal Cell Carcinoma
