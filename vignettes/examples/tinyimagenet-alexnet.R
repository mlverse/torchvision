library(torch)
library(torchvision)
library(magrittr)

dir <- "~/Downloads/tiny-imagenet"

ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      transform_random_resized_crop(size = c(224, 224)) %>%
      transform_random_horizontal_flip() %>%
      transform_normalize(
        mean = c(0.485, 0.456, 0.406),
        std = c(0.229, 0.224, 0.225)
      )
  },
  target_transform = function(x) {
    x <- torch_tensor(x, dtype = torch_long())
    x$squeeze(1)
  }
)

dl <- dataloader(ds, batch_size = 128, shuffle = TRUE)

if (cuda_is_available()) {
  device <- torch_device("cuda")
} else {
  device <- torch_device("cpu")
}

model <- model_alexnet()
model$to(device = device)

optimizer <- optim_adam(model$parameters)
loss_fun <- nn_cross_entropy_loss()

epochs <- 10

for (epoch in 1:50) {

  pb <- progress::progress_bar$new(
    total = length(dl),
    format = "[:bar] :eta Loss: :loss"
  )
  l <- c()

  for (b in enumerate(dl)) {
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = device))
    loss <- loss_fun(output, b[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    l <- c(l, loss$item())
    pb$tick(tokens = list(loss = mean(l)))
  }

  cat(sprintf("Loss at epoch %d: %3f\n", epoch, mean(l)))
}

