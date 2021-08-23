# Packages ----------------------------------------------------------------
library(torch)
library(torchvision)


# Datasets and loaders ----------------------------------------------------

dir <- "~/Downloads/mnist" #caching directory

train_ds <- mnist_dataset(
  dir,
  download = TRUE,
  transform = transform_to_tensor
)

test_ds <- mnist_dataset(
  dir,
  train = FALSE,
  transform = transform_to_tensor
)

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
test_dl <- dataloader(test_ds, batch_size = 32)


# Buildifng the network ---------------------------------------------------

net <- nn_module(
  "Net",
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, 3, 1)
    self$conv2 <- nn_conv2d(32, 64, 3, 1)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- nnf_relu(x)
    x <- self$conv2(x)
    x <- nnf_relu(x)
    x <- nnf_max_pool2d(x, 2)
    x <- self$dropout1(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout2(x)
    output <- self$fc2(x)
    output
  }
)

model <- net()

# ove model to cuda if it's available
device <- if(cuda_is_available()) "cuda" else "cpu"
model$to(device = device)

# Training loop -----------------------------------------------------------

optimizer <- optim_sgd(model$parameters, lr = 0.01)

epochs <- 10
for (epoch in 1:10) {

  pb <- progress::progress_bar$new(
    total = length(train_dl),
    format = "[:bar] :eta Loss: :loss"
  )

  train_losses <- c()
  test_losses <- c()

  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- model(b[[1]]$to(device = device))
    loss <- nnf_cross_entropy(output, b[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
    pb$tick(tokens = list(loss = mean(train_losses)))
  })

  with_no_grad({
    coro::loop(for (b in test_dl) {
      model$eval()
      output <- model(b[[1]]$to(device = device))
      loss <- nnf_cross_entropy(output, b[[2]]$to(device = device))
      test_losses <- c(test_losses, loss$item())
      model$train()
    })
  })

  cat(sprintf("Loss at epoch %d [Train: %3f] [Test: %3f]\n",
              epoch, mean(train_losses), mean(test_losses)))
}

