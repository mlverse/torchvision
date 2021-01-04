
# Packages ----------------------------------------------------------------

library(torch)
library(torchvision)

device <- if (cuda_is_available()) "cuda" else "cpu"
num_steps <- 4000
content_weight <- 1e-5
style_weight <- 1e2


# Network definition ------------------------------------------------------

content_loss <- function(content, style) {
  nnf_mse_loss(content, style)
}

gram_matrix <- function(input) {
  size <- input$size()
  features <- input$view(c(size[1] * size[2], size[3] * size[4]))
  G <- torch_mm(features, features$t())
  # we 'normalize' the values of the gram matrix
  # by dividing by the number of element in each feature maps.
  G$div(prod(size))
}

style_loss <- function(content, style) {
  C <- gram_matrix(content)
  S <- gram_matrix(style)
  nnf_mse_loss(C, S)
}

cnn <- model_vgg19(pretrained = TRUE)$features$to(device = device)
cnn$eval()

# we create an nn_module that does the same as the sequential container but
# returns the results of all convolutions. we also replace inplace operations
# for copy on modify ones
features <- nn_module(
  initialize = function(cnn) {
    self$cnn <- cnn
  },
  forward = function(input) {
    conv_outs <- list()
    for (i in seq_along(self$cnn)) {
      layer <- self$cnn[[i]]

      if (inherits(layer, "nn_relu"))
        input <- nnf_relu(input)
      else
        input <- layer(input)

      if (inherits(layer, "nn_conv2d"))
        conv_outs[[length(conv_outs) + 1]] <- input

    }
    conv_outs
  }
)

model <- features(cnn)

# Loading images ----------------------------------------------------------

norm_mean <- c(0.485, 0.456, 0.406)
norm_std <- c(0.229, 0.224, 0.225)

normalize <- function(img) {
  transform_normalize(img, norm_mean, norm_std)
}

denormalize <- function(img) {
  transform_normalize(img, -norm_mean/norm_std, 1/norm_std)
}

load_image <- function(path) {
  x <- jpeg::readJPEG(path) %>%
    transform_to_tensor() %>%
    transform_resize(c(512, 512))
  x <- x[newaxis,..]
  x <- normalize(x)
  x$to(device = device)
}

style_img <- load_image("vignettes/examples/assets/picasso.jpg")
content_img <- load_image("vignettes/examples/assets/dancing.jpg")
content_img <- content_img$requires_grad_(TRUE)


# Optimization ------------------------------------------------------------

optimizer <- optim_adam(content_img, lr = 1)
lr_scheduler <- lr_step(optimizer, 100, 0.96)

for (step in seq_len(num_steps)) {

  gc() # we have to call gc otherwise R tensors are not disposed.

  optimizer$zero_grad()

  content_features <- model(content_img)
  style_features <- model(style_img)

  # compute the content loss
  l_content <- content_weight * content_loss(content_features[[4]], style_features[[4]])

  # compute the style loss
  l_style <- torch_tensor(0, device = device)
  for (i in 1:5) {
    l_style <- l_style + style_loss(content_features[[i]], style_features[[i]])
  }
  l_style <- style_weight * l_style

  # compute the final loss
  loss <- l_content + l_style

  loss$backward()

  # optimization step
  optimizer$step()
  lr_scheduler$step()

  if (step %% 100 == 0)
    cat(
      "[Step: ", step, "] ",
      "Loss: ", loss$item(),
      " Content loss: ", l_content$item(),
      "Style loss: ", l_style$item(),
      "\n"
    )

}

# Visualize the final img
im <- denormalize(content_img)[1,..]$
  permute(c(2, 3, 1))$
  to(device = "cpu")$
  clamp(0,1) %>% # make it [0,1]
  as.array()

plot(as.raster(im))



