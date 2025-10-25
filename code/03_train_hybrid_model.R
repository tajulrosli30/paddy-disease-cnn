# ============================================================
# Script: 03_train_hybrid_model.R
# Purpose: Train Hybrid CNN (feature concatenation + SE attention)
# ============================================================

source("./code/01_setup_and_data.R")

# Base feature extractors
input_layer <- layer_input(shape=c(img_height, img_width, 3))
base_models <- list(
  resnet50     = application_resnet50(include_top=FALSE, weights="imagenet"),
  densenet121  = application_densenet121(include_top=FALSE, weights="imagenet"),
  inceptionv3  = application_inception_v3(include_top=FALSE, weights="imagenet"),
  basic_cnn    = keras_model_sequential() %>%
    layer_conv_2d(32, 3, activation="relu", input_shape=c(img_height,img_width,3)) %>%
    layer_max_pooling_2d(2) %>%
    layer_conv_2d(64, 3, activation="relu") %>%
    layer_max_pooling_2d(2) %>%
    layer_conv_2d(128, 3, activation="relu") %>%
    layer_max_pooling_2d(2)
)

# Freeze base layers
lapply(base_models, freeze_weights)

# Extract pooled outputs
outputs <- lapply(base_models, function(m) {
  m(input_layer) %>% layer_global_average_pooling_2d()
})

# SE Attention Block
se_block <- function(x, reduction=8L) {
  ch <- k_int_shape(x)[[2]]
  s <- layer_dense(x, units=max(1L,ch %/% reduction), activation="relu")
  s <- layer_dense(s, units=ch, activation="sigmoid")
  layer_multiply(list(x,s))
}

merged <- layer_concatenate(outputs) %>%
  se_block() %>%
  layer_dense(128, activation="relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(6, activation="softmax")

hybrid_model <- keras_model(input_layer, merged)

# Initial training (frozen)
history_hybrid <- fit_and_save(hybrid_model, "hybrid_cnn", "initial")

# Fine-tuning last 20 layers of TL models
lapply(base_models[1:3], function(m) {
  unfreeze_weights(m, from=length(m$layers)-20)
})

hybrid_model %>% compile(
  optimizer=optimizer_adam(1e-5),
  loss=focal_loss(gamma=2, alpha=0.25), metrics="accuracy"
)

history_hybrid_ft <- hybrid_model %>% fit(
  train_generator,
  validation_data=validation_generator,
  epochs=10,
  steps_per_epoch=steps_per_epoch,
  validation_steps=validation_steps,
  callbacks=list(early_stop)
)

save_model_hdf5(hybrid_model, file.path(output_dir, "hybrid_cnn_finetuned.h5"))
write_csv(as.data.frame(history_hybrid_ft), file.path(output_dir, "hybrid_cnn_finetuned_history.csv"))
