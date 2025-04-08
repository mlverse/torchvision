test_that("Training runs without errors", {
  data <- load_spam_data()
  datasets <- prepare_datasets(data)

  train_dl <- torch::dataloader(datasets$train, batch_size = 64, shuffle = TRUE)
  test_dl <- torch::dataloader(datasets$test, batch_size = 64)

  model <- create_model()
  trained_model <- train_model(model, train_dl, test_dl, epochs = 1)

  expect_true(inherits(trained_model, "nn_module"))
})

test_that("Evaluation returns a confusion matrix and valid metrics", {
  data <- load_spam_data()
  datasets <- prepare_datasets(data)

  train_dl <- torch::dataloader(datasets$train, batch_size = 64, shuffle = TRUE)
  test_dl <- torch::dataloader(datasets$test, batch_size = 64)

  model <- create_model()
  trained_model <- train_model(model, train_dl, test_dl, epochs = 1)

  results <- evaluate_model(trained_model, test_dl)

  expect_true(is.list(results))
  expect_true("confusion" %in% names(results))
  expect_true("accuracy" %in% names(results))
  expect_type(results$accuracy, "double")
})

library(testthat)
library(torchvision)

if (Sys.getenv("TORCH_TEST", unset = 0) == 1)
  test_check("torchvision")

