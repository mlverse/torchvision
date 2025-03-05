library(testthat)
library(torch)
library(arrow)

# Load dataset loader (if needed)
# source(file.path("R", "caltech101_dataset_loader.R"))  # Only if it's not part of a package

temp_root <- tempfile(fileext = "/")

test_that("caltech101_dataset downloads correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")

  # Expect an error if the file is missing and download = FALSE
  expect_error(
    caltech101_dataset(root = temp_root, download = FALSE),
    "Caltech-101 Parquet file not found",
    label = "Dataset should fail if not previously downloaded"
  )

  # Download the dataset
  expect_no_error(
    ds <- caltech101_dataset(root = temp_root, download = TRUE),
    label = "Dataset should download successfully"
  )

  # Check if the file exists
  parquet_file <- file.path(temp_root, "train-00000-of-00001.parquet")
  expect_true(file.exists(parquet_file), info = "Parquet file should exist after download")

  # Check if the file has a non-zero size
  expect_gt(file.size(parquet_file), 0, info = "Parquet file should not be empty")
})
