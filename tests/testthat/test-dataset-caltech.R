library(testthat)
library(torch)
library(jpeg)

temp_root <- tempfile(fileext = "/")

test_that("caltech101_dataset downloads and extracts correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("jpeg")
  skip_if_offline()

  # Test download
  expect_no_error(
    ds <- caltech101_dataset(root = temp_root, download = TRUE),
    info = "Dataset should download successfully"
  )

  # Verify the zip file was downloaded
  zip_file <- file.path(temp_root, "caltech-101.zip")
  expect_true(file.exists(zip_file),
              label = "Zip file should exist after download")
  expect_gt(file.size(zip_file), 0,
            label = "Zip file should not be empty")

  # Verify the main tar.gz was extracted
  tar_gz_file <- file.path(temp_root, "caltech-101", "101_ObjectCategories.tar.gz")
  expect_true(file.exists(tar_gz_file),
              label = "Main tar.gz should exist after unzip")

  # Verify the final image directory exists
  image_dir <- file.path(temp_root, "caltech-101", "101_ObjectCategories")
  expect_true(dir.exists(image_dir),
              label = "Image directory should exist after full extraction")

  # Verify we have some images
  image_files <- list.files(image_dir, pattern = "\\.jpg$", recursive = TRUE)
  expect_gt(length(image_files), 0,
            label = "Should find some JPG images after extraction")
})

test_that("caltech101_dataset loads samples correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("jpeg")
  skip_if_not(dir.exists(file.path(temp_root, "caltech-101", "101_ObjectCategories")),
              "Dataset not downloaded")

  ds <- caltech101_dataset(root = temp_root, download = FALSE)

  # Test sample loading
  sample <- ds[1]

  # Verify sample structure
  expect_true(is.list(sample), label = "Sample should be a list")
  expect_named(sample, c("x", "y"), ignore.order = TRUE)

  # Check image is array
  expect_true(is.array(sample$x), label = "Image should be array without transform")

  # Check image dimensions (H x W x C)
  img_dims <- dim(sample$x)
  expect_equal(length(img_dims), 3, label = "Array images should be H x W x C")
  expect_equal(img_dims[3], 3, label = "Should have 3 color channels")

  # Check label is tensor
  expect_true(inherits(sample$y, "torch_tensor"),
              label = "Label should be torch tensor")
})

test_that("caltech101_dataset transforms work correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("jpeg")
  skip_if_not(dir.exists(file.path(temp_root, "caltech-101", "101_ObjectCategories")),
              "Dataset not downloaded")

  # Test transform that converts to tensor and normalizes
  transform <- function(x) {
    torch::torch_tensor(x)$permute(c(3, 1, 2))$div(255) # HWC to CHW and normalize
  }

  ds <- caltech101_dataset(
    root = temp_root,
    transform = transform
  )

  sample <- ds[1]

  # Verify transformed output
  expect_true(inherits(sample$x, "torch_tensor"),
              label = "Transformed image should be tensor")
  expect_equal(dim(sample$x)[1], 3,
               label = "Transformed image should be CHW format")

  # Verify normalization
  expect_lte(max(as.array(sample$x)), 1,
             label = "Normalized values should be <= 1")
  expect_gte(min(as.array(sample$x)), 0,
             label = "Normalized values should be >= 0")
})

# Clean up
unlink(temp_root, recursive = TRUE)
