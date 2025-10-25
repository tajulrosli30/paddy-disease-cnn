# 🍃 Paddy Disease CNN

This repository contains reproducible code and pretrained models for the paper:

**"Towards Deployable and Explainable Deep Learning Models for Paddy Leaf Disease Classification in R: A Comparative Study of CNN Architectures with SHAP and LIME"**  
(Submitted to *Expert Systems with Applications*, Revision: Oct 2025)

## Contact

Dr. Tajul Rosli Razak
Universiti Teknologi MARA (UiTM), Malaysia
Email: tajulrosli@uitm.edu.my


## 📂 Repository Structure

paddy-disease-cnn/ 
1. code/                 # R scripts for model training & evaluation
2. models/               # Pretrained models (.h5) via Git LFS
3. results/              # Evaluation outputs & predictions
4. README.md             # This file

---

## 📊 Dataset

The dataset used in this project is sourced from Paddy Disease Clean Solution:  
https://www.kaggle.com/code/abdmental01/paddy-disease-clean-solution

After downloading:
data/
1. TRAIN_DATASET/
2. VALIDATION_DATASET/

Paths can be updated inside the R scripts.

---

## ✅ Pretrained Models

Models can be downloaded directly from the `models/` folder:

- resnet50_finetuned.h5  
- densenet121_finetuned.h5  
- inceptionv3_finetuned.h5  
- hybrid_simple_preds.rds (saved predictions)

These allow users to reproduce results without retraining.

---

## ▶️ How to Run the Code

Install required R packages:

```r
install.packages(c("keras", "tensorflow", "caret", "readr"))


