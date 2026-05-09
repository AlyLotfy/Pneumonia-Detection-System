# Pneumonia Detection System

**A deep-learning diagnostic aid that classifies chest X-rays as normal or pneumonia — custom CNN reaches 97.2 % test accuracy, outperforming a frozen ResNet50 baseline on this dataset.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](#)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](#)

---

## Overview

The Pneumonia Detection System classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using deep learning. Two architectures are built and compared:

1. A **custom CNN** trained from scratch — 3 convolutional blocks with max-pooling, followed by a dense head with dropout.
2. **ResNet50 transfer learning** — pretrained ImageNet weights used as a frozen feature extractor, with a custom classification head.

The custom CNN wins on this dataset despite being simpler: it achieves **97.2 % test accuracy** vs ResNet50's **69 %** (with frozen layers). The comparison illustrates why "use a pretrained model" is not always the right answer — frozen ResNet50 features learned on natural images do not transfer as cleanly to medical X-ray textures without fine-tuning.

> **Disclaimer.** This is an academic project. It has not been validated for clinical use and must not be deployed for diagnosis without regulatory review.

---

## Dataset

- **Source:** [Chest X-ray Pneumonia — Paul Mooney (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** 5,216 images (train) + 624 (test) + 16 (val)
- **Classes:** `NORMAL` (1,341 train), `PNEUMONIA` (3,875 train)
- **Challenge:** 3:1 class imbalance toward pneumonia cases

---

## Models

### Custom CNN

```
Input: (128, 128, 3)
  Conv2D(256, 3×3, ReLU) → MaxPool
  Conv2D(128, 3×3, ReLU) → MaxPool
  Conv2D(64,  3×3, ReLU) → MaxPool
  Flatten
  Dense(128, ReLU) → Dropout(0.5)
  Dense(2, sigmoid)
```

Trained for 10 epochs with Adam optimizer and `sparse_categorical_crossentropy` loss. No augmentation — the CNN converges cleanly on the raw normalized images.

### ResNet50 (Transfer Learning)

- Base: `ResNet50(weights='imagenet', include_top=False)` — all layers frozen
- Head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dense(1, sigmoid)`
- Input resized to 224×224 for ResNet compatibility
- Trained with `ImageDataGenerator` augmentation (rotation, zoom, horizontal flip)
- Callbacks: `ModelCheckpoint`, `EarlyStopping(patience=5)`, `ReduceLROnPlateau`

---

## Results

### Custom CNN (primary model)

| Metric | Score |
| --- | --- |
| **Test Accuracy** | **97.2 %** |
| Training Accuracy (epoch 10) | 98.5 % |
| Validation Accuracy | 96.7 % |
| Precision (macro avg) | 97 % |
| Recall (macro avg) | 96 % |
| F1-score (macro avg) | 97 % |

Confusion matrix on 1,044 test images:

|  | Pred: NORMAL | Pred: PNEUMONIA |
| --- | --- | --- |
| **True: NORMAL** | 274 | 16 |
| **True: PNEUMONIA** | 13 | 741 |

### ResNet50 (frozen baseline)

| Metric | Score |
| --- | --- |
| Test Accuracy | 69 % |

ResNet50 with frozen weights underperforms significantly. Fine-tuning the upper ResNet layers would close most of this gap, but was not explored here.

---

## Key Takeaway

On this pediatric X-ray dataset, a purpose-built CNN trained from scratch is substantially better than a frozen ImageNet feature extractor. The custom CNN also trains in ~15 ms per image at inference vs ~28 ms for ResNet50.

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model definition, training, evaluation
- **PyTorch / torchvision** — ResNet50 model loading
- **NumPy / pandas** — data handling
- **OpenCV** — image reading and resizing
- **scikit-learn** — train/test split, classification report, confusion matrix
- **matplotlib** — training curves and prediction visualization

---

## Quick Start

```bash
git clone https://github.com/AlyLotfy/Pneumonia-Detection-System.git
cd Pneumonia-Detection-System

pip install tensorflow torch torchvision numpy pandas opencv-python scikit-learn matplotlib jupyter

# 1. Download the dataset from Kaggle (chest-xray-pneumonia) and place it under ./chest_xray/
# 2. Launch the notebook
jupyter notebook
```

---

## Future Work

- Fine-tune upper ResNet50 layers to close the gap with the custom CNN
- Grad-CAM visualizations to show which lung regions drive each prediction
- External validation on adult and COVID-era chest X-ray datasets
- Multi-label extension: distinguish bacterial vs. viral pneumonia

---

## References

- Kermany et al., 2018. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell.
- He et al., 2016. *Deep Residual Learning for Image Recognition.* CVPR.
- Selvaraju et al., 2017. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.

---

## License

Academic project, AAST College of Artificial Intelligence. Educational use only — not validated for clinical deployment.
