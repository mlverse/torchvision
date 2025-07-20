test_that("places365_dataset loads and returns valid items", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset(
    split = "val",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_s3_class(ds, "places365_dataset")
  expect_length(ds, 36500)

  item <- ds[1]

  expect_type(item, "list")
  expect_named(item, c("x", "y"))

  # Check image
  expect_tensor(item$x)
  expect_equal(item$x$ndim, 3)
  expect_equal(item$x$shape[1], 3)
  expect_gt(item$x$shape[2], 0)
  expect_gt(item$x$shape[3], 0)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_gte(torch::torch_min(item$x)$item(), 0)
  expect_lte(torch::torch_max(item$x)$item(), 1)

  # Check label
  expect_type(item$y, "integer")
  expect_gte(item$y, 1)
  expect_lte(item$y, 365)
})

test_that("places365_dataset train split loads an item", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset(
    split = "train",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_gt(length(ds), 0)

  item <- ds[1]
  expect_type(item, "list")
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_type(item$y, "integer")
  expect_gte(item$y, 1)
  expect_lte(item$y, 365)
})

test_that("places365 categories are correctly mapped", {
  ann_path <- file.path(tempdir(), "places365", "categories_places365.txt")
  skip_if_not(file.exists(ann_path), message = "Category file missing")

  categories <- readLines(ann_path, warn = FALSE)
  expect_length(categories, 365)

  ds <- places365_dataset(
    split = "val",
    download = FALSE,
    transform = transform_to_tensor
  )

  label <- ds[1]$y
  expect_type(label, "integer")
  expect_lte(label, length(categories))
})

test_that("places365_dataset test split returns image only", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset(
    split = "test",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_gt(length(ds), 0)

  item <- ds[1]
  expect_type(item, "list")
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_true(is.na(item$y))
})

# Tests for the high-resolution Places365 variant

test_that("places365_dataset_large loads and returns valid items", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset_large(
    split = "val",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_s3_class(ds, "places365_dataset_large")
  expect_length(ds, 36500)

  item <- ds[1]

  expect_type(item, "list")
  expect_named(item, c("x", "y"))

  # Check image tensor has 3 channels and positive dims
  expect_tensor(item$x)
  expect_equal(item$x$ndim, 3)
  expect_equal(item$x$shape[1], 3)
  expect_gt(item$x$shape[2], 0)
  expect_gt(item$x$shape[3], 0)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_gte(torch::torch_min(item$x)$item(), 0)
  expect_lte(torch::torch_max(item$x)$item(), 1)

  # Check label range
  expect_type(item$y, "integer")
  expect_gte(item$y, 1)
  expect_lte(item$y, 365)
})

test_that("places365_dataset_large train split loads an item", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset_large(
    split = "train",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_gt(length(ds), 0)

  item <- ds[1]
  expect_type(item, "list")
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_type(item$y, "integer")
  expect_gte(item$y, 1)
  expect_lte(item$y, 365)
})

test_that("places365_dataset_large test split returns image only", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset_large(
    split = "test",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_gt(length(ds), 0)

  item <- ds[1]
  expect_type(item, "list")
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_dtype(item$x, torch::torch_float())
  expect_true(is.na(item$y))
})

test_that("places365_dataset_large can be batched with resize_collate_fn", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- places365_dataset_large(
    split = "val",
    download = TRUE,
    transform = transform_to_tensor
  )

  dl <- torch::dataloader(
    dataset = ds,
    batch_size = 2,
    collate_fn = torchvision:::resize_collate_fn(c(256, 256))
  )

  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_tensor_shape(batch$x, c(2, 3, 256, 256))
  expect_tensor_dtype(batch$x, torch::torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y, 2)
  expect_tensor_dtype(batch$y, torch::torch_long())
})
