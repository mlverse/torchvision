test_that("eurosat_dataset API handles splits correctly via API", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  # Test train split
  train_ds <- eurosat_dataset(root = "./data/eurosat", split = "train", download = TRUE)
  expect_true(length(train_ds) > 0, info = "Train dataset should have a non-zero length")
  
  # Test validation split
  validation_ds <- eurosat_dataset(root = "./data/eurosat", split = "validation", download = TRUE)
  expect_true(length(validation_ds) > 0, info = "Validation dataset should have a non-zero length")
  
  # Test test split
  test_ds <- eurosat_dataset(root = "./data/eurosat", split = "test", download = TRUE)
  expect_true(length(test_ds) > 0, info = "Test dataset should have a non-zero length")
  
  # Test data content
  sample <- train_ds[1]
  expect_true(!is.null(sample$x), info = "Image should not be null")
  expect_true(!is.null(sample$y), info = "Label should not be null")
})
