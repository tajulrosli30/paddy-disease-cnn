# ============================================================
# Script: 02_train_single_models.R
# Purpose: Train & fine-tune three pre-trained CNN architectures
# ============================================================

source("./code/01_setup_and_data.R")

# General training function for TL models
train_finetune_model <- function(base_fn, name) {
  
  base_model <- base_fn(include_top=FALSE, weights="imagenet",
                        input_shape=c(img_height,img_width,3))
  freeze_weights(base_model)
  
  model <- keras_model_sequential() %>%
    base_model %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(128, activation="relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(6, activation="softmax")
  
  # Initial training
  hist_init <- fit_and_save(model, name, "initial")
  
  # Fine-tuning → last 20 layers trainable
  unfreeze_weights(base_model, from=length(base_model$layers)-20)
  model %>% compile(
    optimizer=optimizer_adam(1e-5),
    loss=focal_loss(gamma=2, alpha=0.25),
    metrics="accuracy"
  )
  hist_ft <- model %>% fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10, steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=list(early_stop)
  )
  
  save_model_hdf5(model, file.path(output_dir, paste0(name, "_finetuned.h5")))
  write_csv(as.data.frame(hist_ft), file.path(output_dir, paste0(name, "_finetuned_history.csv")))
  
  return(list(initial=hist_init, finetune=hist_ft))
}

# Train 3 transfer learning models
hist_resnet   <- train_finetune_model(application_resnet50, "resnet50")
hist_densenet <- train_finetune_model(application_densenet121, "densenet121")
hist_inception<- train_finetune_model(application_inception_v3, "inceptionv3")
