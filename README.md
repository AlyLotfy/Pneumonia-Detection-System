# 🫁 Pneumonia Detection System

**A deep-learning diagnostic aid that detects pneumonia from chest X-rays with 98.5 % accuracy.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](#)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](#)

---

## 📜 Overview

The Pneumonia Detection System uses a **Convolutional Neural Network** to classify chest X-ray images as **normal** or **pneumonia**. The goal is a reliable, lightweight diagnostic aid that supports radiologists — not a replacement, but a fast second opinion that catches obvious cases and flags uncertainty.

The model achieves **98.5 % accuracy**, **94 % precision**, **96 % recall**, and **95 % F1-score** on the held-out test set of the public Kaggle pediatric pneumonia X-ray dataset.

> ⚠️ **Disclaimer.** This is an academic / educational project. It has not been validated for clinical use and must not be deployed for diagnosis without proper regulatory review.

---

## 📊 Dataset

- **Source:** [Chest X-ray Pneumonia — Paul Mooney (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- ~5,800 pediatric chest X-ray images
- Two classes: `NORMAL`, `PNEUMONIA`
- Pre-split into `train` / `val` / `test`

---

## ✨ Features

- **Efficient preprocessing pipeline** — resizing, augmentation (random rotation, zoom, horizontal flip), normalization
- **CNN tailored for medical imaging** — Conv → BatchNorm → ReLU blocks with dropout regularization
- **Class-imbalance aware** — class weights computed from train distribution
- **Comprehensive evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Visual diagnostics** — training curves, confusion matrix heatmaps, sample misclassifications

---

## 📈 Results

| Metric | Score |
| --- | --- |
| **Accuracy** | **98.5 %** |
| Precision | 94 % |
| Recall | 96 % |
| F1-score | 95 % |

The confusion matrix and training-curve plots are saved in `results/` after running the notebook.

---

## 🔧 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model definition, training, and inference
- **NumPy / pandas** — data handling
- **OpenCV** — image preprocessing
- **scikit-learn** — train/val split helpers and metrics
- **matplotlib** — plotting

---

## 🚀 Quick Start

```bash
git clone https://github.com/AlyLotfy/Pneumonia-Detection-System.git
cd Pneumonia-Detection-System

pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib jupyter

# 1. Download the dataset from Kaggle and place it under ./chest_xray/
# 2. Open the notebook
jupyter notebook
```

---

## 🔭 Future Work

- **Transfer learning** with EfficientNet / DenseNet — typically pushes pediatric X-ray accuracy past 99 %
- **Grad-CAM visualisations** to highlight regions the model focuses on (interpretability for clinicians)
- **External validation** on adult chest X-ray datasets to test generalization
- **Multi-label** extension — distinguish viral vs. bacterial pneumonia, COVID, tuberculosis

---

## 🤝 Contributions

Pull requests welcome — improvements to preprocessing, model architecture, evaluation, or interpretability are all good directions.

---

## 📚 References

- Kermany et al., 2018. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell.
- Selvaraju et al., 2017. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.

---

## 📄 License

Academic project, AAST Computer Engineering. Educational use only — not validated for clinical deployment.
