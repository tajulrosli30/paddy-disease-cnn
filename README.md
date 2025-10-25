# Paddy Disease CNN

This repository contains reproducible code and pretrained models for the paper:

**"Towards Deployable and Explainable Deep Learning Models for Paddy Leaf Disease Classification in R: A Comparative Study of CNN Architectures with SHAP and LIME"**  
(Submitted to *Expert Systems with Applications*, Revision: Oct 2025)

---

## Repository Structure
paddy-disease-cnn/
│
├── code/ # R scripts for model training & evaluation
├── models/ # Pretrained models (.h5) via Git LFS
├── results/ # Evaluation outputs & predictions
└── README.md # This file

---

## Dataset Access

The dataset used in this study can be downloaded from Kaggle:

🔗 https://www.kaggle.com/datasets/arjuntejaswi/plant-village](https://www.kaggle.com/code/abdmental01/paddy-disease-clean-solution

After downloading, place the dataset as:
data/
├── TRAIN_DATASET/
└── VALIDATION_DATASET/


Paths can be changed in the R scripts if needed.

---

## Pretrained Models

Download pretrained models from the `models/` folder.

Models available:
- `hybrid_cnn_finetuned.h5` ✅ Best model
- `resnet50_finetuned.h5`
- `densenet121_finetuned.h5`
- `inceptionv3_finetuned.h5`
- `hybrid_simple_preds.rds` (prediction output)

These allow users to reproduce results **without retraining**.

---

## How to Run

Install packages as needed:

```r
install.packages(c("keras", "tensorflow", "caret", "readr"))

source("code/01_setup_and_data.R")
source("code/02_train_single_models.R")
source("code/03_train_hybrid_model.R")
source("code/04_evaluation_and_cv.R")


