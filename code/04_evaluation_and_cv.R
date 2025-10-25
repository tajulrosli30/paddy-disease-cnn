# ============================================================
# Script: 04_evaluation_and_cv.R
# Purpose: Per-class metrics & 5-Fold CV for Hybrid CNN
# ============================================================

source("./code/01_setup_and_data.R")

# Load model
hybrid_model <- load_model_hdf5("./results/hybrid_cnn_finetuned.h5")

# Validation Predictions
pred <- hybrid_model %>% predict(validation_generator, steps=validation_steps)
y_pred <- apply(pred, 1, which.max)
y_true <- validation_generator$classes[1:length(y_pred)] + 1

cm <- confusionMatrix(factor(y_pred), factor(y_true))
print(cm)

macroF1 <- mean(cm$byClass[ , "F1"], na.rm=TRUE)
cat("Macro F1:", round(macroF1,4), "\n")

# ✅ Optional - 5-Fold Cross Validation Script Placeholder
# (Users can enable if dataset split is available)
