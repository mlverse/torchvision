library(testthat)
library(arrow)

# Define the path to the dataset inside the project
dataset_path <- file.path("data", "train.parquet")

# Test that the dataset loads correctly from local storage
test_that("Caltech-101 dataset loads correctly from local storage", {
  expect_true(file.exists(dataset_path), info = "Dataset file not found in data/ folder.")

  # Load the dataset
  dataset <- read_parquet(dataset_path)

  # Ensure required columns exist
  expect_true("image" %in% colnames(dataset) || "filename" %in% colnames(dataset),
              info = "Dataset must have either 'image' or 'filename' column.")
  expect_true("label" %in% colnames(dataset), info = "Column 'label' not found in dataset.")

  # Check that dataset has some rows
  expect_gt(nrow(dataset), 0, info = "Dataset should not be empty.")

  # Print first few rows (for debugging purposes)
  print(head(dataset))
})
