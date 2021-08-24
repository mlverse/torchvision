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
    self$fc1 <- nn_linear(784, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2()
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

