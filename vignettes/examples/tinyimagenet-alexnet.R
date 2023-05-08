# Packages ----------------------------------------------------------------

library(torch)
library(torchvision)


# Datasets ----------------------------------------------------------------

dir <- "~/Downloads/tiny-imagenet"

device <- if(cuda_is_available()) "cuda" else "cpu"

to_device <- function(x, device) {
  x$to(device = device)
}

train_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      to_device(device) %>%
      transform_resize(c(64, 64))
  }
)

valid_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  split = "val",
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      to_device(device) %>%
      transform_resize(c(64,64))
  }
)

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)


# Model -------------------------------------------------------------------

model <- model_alexnet(pretrained = FALSE, num_classes = length(train_ds$classes))
model$to(device = device)

optimizer <- optim_adagrad(model$parameters, lr = 0.005)
scheduler <- lr_step(optimizer, step_size = 1, 0.95)
loss_fn <- nn_cross_entropy_loss()


# Training loop -----------------------------------------------------------

train_step <- function(batch) {
  optimizer$zero_grad()
  output <- model(batch[[1]]$to(device = device))
  loss <- loss_fn(output, batch[[2]]$to(device = device))
  loss$backward()
  optimizer$step()
  loss
}

valid_step <- function(batch) {
  model$eval()
  pred <- model(batch[[1]]$to(device = device))
  pred <- torch_topk(pred, k = 5, dim = 2, TRUE, TRUE)[[2]]
  pred <- pred$to(device = torch_device("cpu"))
  correct <- batch[[2]]$view(c(-1, 1))$eq(pred)$any(dim = 2)
  model$train()
  correct$to(dtype = torch_float32())$mean()$item()
}

for (epoch in 1:50) {

  pb <- progress::progress_bar$new(
    total = length(train_dl),
    format = "[:bar] :eta Loss: :loss"
  )

  l <- c()
  coro::loop(for (b in train_dl) {
    loss <- train_step(b)
    l <- c(l, loss$item())
    pb$tick(tokens = list(loss = mean(l)))
  })

  acc <- c()
  with_no_grad({
    coro::loop(for (b in valid_dl) {
      accuracy <- valid_step(b)
      acc <- c(acc, accuracy)
    })
  })

  scheduler$step()
  cat(sprintf("[epoch %d]: Loss = %3f, Acc= %3f \n", epoch, mean(l), mean(acc)))
}
