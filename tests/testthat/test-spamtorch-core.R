library(testthat)
library(spamtorch)
library(torch)

test_that("Data loads correctly", {
  data <- load_spam_data()
  expect_type(data, "list")
  expect_true(!is.null(data$features))
  expect_true(!is.null(data$labels))
  expect_equal(ncol(data$features), 57)
  expect_equal(length(data$labels), nrow(data$features))
})

test_that("Datasets are created and scaled properly", {
  data <- load_spam_data()
  datasets <- prepare_datasets(data, train_ratio = 0.8)
  
  expect_type(datasets, "list")
  expect_true("train" %in% names(datasets))
  expect_true("test" %in% names(datasets))
  expect_true("scaler" %in% names(datasets))
})

test_that("Model is created and outputs correct shape", {
  model <- create_model()
  input <- torch_randn(10, 57)  # simulate a batch of 10 samples
  output <- model(input)
  expect_equal(output$size()[[1]], 10)
  expect_equal(output$size()[[2]], 2)
})

test_that("Training runs without errors", {
  data <- load_spam_data()
  datasets <- prepare_datasets(data)
  
  train_dl <- dataloader(datasets$train, batch_size = 64, shuffle = TRUE)
  test_dl <- dataloader(datasets$test, batch_size = 64)
  
  model <- create_model()
  # Run a very short training run for testing purposes.
  trained_model <- train_model(model, train_dl, test_dl, max_epochs = 1, patience = 1)
  expect_s3_class(trained_model, "nn_module")
})

test_that("Evaluation returns a confusion matrix and valid metrics", {
  data <- load_spam_data()
  datasets <- prepare_datasets(data)
  
  train_dl <- dataloader(datasets$train, batch_size = 64, shuffle = TRUE)
  test_dl <- dataloader(datasets$test, batch_size = 64)
  
  model <- create_model()
  trained_model <- train_model(model, train_dl, test_dl, max_epochs = 1, patience = 1)
  results <- evaluate_model(trained_model, test_dl)
  
  expect_true("confusion_matrix" %in% names(results))
  expect_true("accuracy" %in% names(results))
  expect_type(results$accuracy, "double")
})
