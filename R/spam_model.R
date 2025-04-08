#' Load and preprocess the spam dataset
#' @return A list with features and labels
#' @export
load_spam_data <- function() {
  url <- "https://hastie.su.domains/ElemStatLearn/datasets/spam.data"
  tmp <- tempfile()
  utils::download.file(url, tmp)
  data <- utils::read.table(tmp)
  unlink(tmp)

  features <- as.matrix(data[, -58])
  labels <- as.integer(data[, 58])

  list(features = features, labels = labels)
}

#' Create a torch dataset
dataset_wrapper <- function(features, labels) {
  torch::dataset(
    name = "SpamDataset",
    initialize = function() {
      self$x <- torch::torch_tensor(features, dtype = torch::torch_float())
      self$y <- torch::torch_tensor(labels + 1L, dtype = torch::torch_long())  # 1-based for torch
    },
    .getitem = function(i) {
      list(x = self$x[i, ], y = self$y[i])
    },
    .length = function() {
      self$y$size()[[1]]
    }
  )()
}

#' Prepare train/test datasets
#' @param data List returned from load_spam_data
#' @param train_ratio Fraction of data to use for training
#' @return A list with train/test datasets and scaler info
#' @export
prepare_datasets <- function(data, train_ratio = 0.8) {
  set.seed(123)
  idx <- sample(seq_len(nrow(data$features)))
  n_train <- floor(train_ratio * length(idx))

  train_idx <- idx[1:n_train]
  test_idx <- idx[(n_train + 1):length(idx)]

  train_features <- data$features[train_idx, ]
  train_labels <- data$labels[train_idx]

  test_features <- data$features[test_idx, ]
  test_labels <- data$labels[test_idx]

  mean_vals <- colMeans(train_features)
  sd_vals <- apply(train_features, 2, sd)

  train_scaled <- scale(train_features, center = mean_vals, scale = sd_vals)
  test_scaled <- scale(test_features, center = mean_vals, scale = sd_vals)

  list(
    train = dataset_wrapper(train_scaled, train_labels),
    test = dataset_wrapper(test_scaled, test_labels),
    scaler = list(mean = mean_vals, sd = sd_vals)
  )
}

#' Create a neural network model
#' @param input_dim Number of input features
#' @param hidden_dim Hidden layer size
#' @param output_dim Number of output classes
#' @return A torch model
#' @export
create_model <- function(input_dim = 57, hidden_dim = 64, output_dim = 2) {
  torch::nn_module(
    initialize = function() {
      self$fc1 <- torch::nn_linear(input_dim, hidden_dim)
      self$fc2 <- torch::nn_linear(hidden_dim, output_dim)
    },
    forward = function(x) {
      x %>% self$fc1() %>% torch::nnf_relu() %>% self$fc2()
    }
  )()
}

#' Train a neural network model
#' @param model The torch model
#' @param train_dl Training dataloader
#' @param test_dl Test dataloader (not used in training)
#' @param lr Learning rate
#' @param epochs Number of epochs
#' @return Trained model
#' @export
train_model <- function(model, train_dl, test_dl, lr = 0.001, epochs = 10, ...) {
  optimizer <- torch::optim_adam(model$parameters, lr = lr)
  for (epoch in 1:epochs) {
    model$train()
    coro::loop(for (batch in train_dl) {
      optimizer$zero_grad()
      output <- model(batch$x)
      loss <- torch::nnf_cross_entropy(output, batch$y)
      loss$backward()
      optimizer$step()
    })
  }
  return(model)
}

#' Evaluate model on test data
#' @param model The trained model
#' @param test_dl Dataloader for test set
#' @return List with confusion matrix and accuracy
#' @export
evaluate_model <- function(model, test_dl) {
  model$eval()
  all_preds <- c()
  all_labels <- c()
  coro::loop(for (batch in test_dl) {
    output <- model(batch$x)
    preds <- output$argmax(dim = 2)$to(dtype = torch::torch_int())
    all_preds <- c(all_preds, as.integer(preds))
    all_labels <- c(all_labels, as.integer(batch$y))
  })

  table <- table(factor(all_preds, levels = 1:2),
                 factor(all_labels, levels = 1:2))
  acc <- sum(diag(table)) / sum(table)
  list(confusion = table, accuracy = acc)
}
