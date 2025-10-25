# ============================================================
# Paddy Leaf Disease Classification - Main Training Script
# ============================================================

# Required Libraries
library(keras)
library(tensorflow)
library(caret)
library(readr)
library(reticulate)

# ✅ Use default TensorFlow environment
# (This avoids local Windows-specific conda paths)
tf$constant("TensorFlow environment OK")

# ✅ Reduce TensorFlow logs
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

# ✅ Relative Paths for GitHub Reproducibility
train_dir <- "./data/train"
val_dir   <- "./data/val"
output_dir <- "./results"

# Auto-create results directory if not exists
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# 📌 Dataset info for README (not run automatically)
# Dataset: Paddy Doctor / Paddy Disease Classification Dataset (Kaggle)
# Download link:
# https://www.kaggle.com/datasets/vbookshelf/paddy-disease-classification
#
# After download, extract into:
# ./data/train/CLASSNAME/*.jpg
# ./data/val/CLASSNAME/*.jpg

# ============================
# Training Configuration
# ============================
img_height <- 128; img_width <- 128; batch_size <- 16; epochs <- 30

train_datagen <- image_data_generator(
  rescale=1/255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
  shear_range=0.2, zoom_range=0.2, horizontal_flip=TRUE, fill_mode="nearest"
)
val_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(
  train_dir, train_datagen,
  target_size=c(img_height,img_width),
  batch_size=batch_size,
  class_mode="categorical", shuffle=TRUE
)
validation_generator <- flow_images_from_directory(
  val_dir, val_datagen,
  target_size=c(img_height,img_width),
  batch_size=batch_size,
  class_mode="categorical", shuffle=FALSE
)

steps_per_epoch <- floor(train_generator$n / batch_size)
validation_steps <- floor(validation_generator$n / batch_size)

early_stop <- callback_early_stopping(monitor="val_loss", patience=5, restore_best_weights=TRUE)

# ============================
# Focal Loss
# ============================
focal_loss <- function(gamma = 2.0, alpha = 0.25) {
  function(y_true, y_pred) {
    eps <- k_epsilon()
    y_pred <- k_clip(y_pred, eps, 1 - eps)
    ce <- - (y_true * k_log(y_pred))
    pt <- y_true * y_pred + (1 - y_true) * (1 - y_pred)
    fl <- alpha * k_pow(1 - pt, gamma) * ce
    k_sum(fl, axis = -1)
  }
}

# ============================
# Helper Function: Fit & Save
# ============================
fit_and_save <- function(model, name, suffix) {
  model %>% compile(
    optimizer = optimizer_adam(learning_rate=0.001),
    loss = focal_loss(gamma=2, alpha=0.25),
    metrics = "accuracy"
  )
  history <- model %>% fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=list(early_stop)
  )
  
  save_model_hdf5(model, file.path(output_dir, paste0(name, "_", suffix, ".h5")))
  write_csv(as.data.frame(history), file.path(output_dir, paste0(name, "_", suffix, "_history.csv")))
  return(history)
}

# ============================
# Baseline CNN
# ============================
model_basic <- keras_model_sequential() %>%
  layer_conv_2d(32, 3, activation="relu", input_shape=c(img_height,img_width,3)) %>%
  layer_max_pooling_2d(2) %>%
  layer_conv_2d(64, 3, activation="relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_conv_2d(128, 3, activation="relu") %>%
  layer_max_pooling_2d(2) %>%
  layer_flatten() %>%
  layer_dense(128, activation="relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(6, activation="softmax")

history_basic <- fit_and_save(model_basic, "basic_cnn", "initial")
