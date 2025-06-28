temp_root <- withr::local_tempdir()

test_that("eurosat_dataset downloads correctly whatever the split", {
  skip_on_cran()
  skip_if_not_installed("torch")


  expect_error(
    eurosat_dataset(root = tempfile(), split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  )

  expect_is(ds, "dataset", "train should be a dataset")

  extracted_dir <- file.path(temp_root, "eurosat", "images", "2750")
  extracted_dir <- list.dirs(extracted_dir, recursive = FALSE, full.names = TRUE)
  # Extracted data folder have one folder per category
  expect_length(extracted_dir, 10)

  image_files <- list.files(extracted_dir, pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)
  expect_gte(length(image_files), 16200, "Image files should be present in the extracted directory")

  train_ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  # Train dataset should have exactly 16200 samples and reuse existing folder
  expect_length(train_ds, 16200)

  val_ds <- eurosat_dataset(root = temp_root, split = "val", download = TRUE)
  # Validation dataset should have exactly 5400 samples
  expect_length(val_ds, 5400)

  test_ds <- eurosat_dataset(root = temp_root, split = "test", download = TRUE)
  # Test dataset should have exactly 5400 samples
  expect_length(test_ds, 5400)

})

test_that("dataloader from eurosat_dataset gets torch tensors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  ds <- eurosat_dataset(root = temp_root, split = "train", download = FALSE, transform = transform_to_tensor)
  dl <- torch::dataloader(ds, batch_size = 10)
  # 16,2k turns into 1620 batches of 10
  expect_length(dl, 1620)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  # Check shape, dtype, and values on X
  expect_tensor_shape(i[[1]], c(10, 3, 64, 64))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_tensor_shape(i[[2]], 10)
  expect_tensor_dtype(i[[2]], torch_long())
  expect_named(i, c("x", "y"))})


test_that("eurosat100_dataset derivatives download and prepare correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")


  expect_error(
    eurosat100_dataset(root = tempfile(), split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    ds_100 <- eurosat100_dataset(root = temp_root, split = "val", download = TRUE)
  )

  dl <- torch::dataloader(ds_100, batch_size = 10)
  # 20 turns into 2 batches of 10
  expect_length(dl, 2)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)

  # Check shape, dtype, and values on X
  expect_tensor_shape(i[[1]], c(10, 13, 64, 64))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_tensor_shape(i[[2]], 10)
  expect_tensor_dtype(i[[2]], torch_long())
  expect_named(i, c("x", "y"))})

test_that("eurosat_all_bands_dataset derivatives download and prepare correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")


  expect_error(
    eurosat_all_bands_dataset(root = tempfile(), split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    ds_all <- eurosat_all_bands_dataset(root = temp_root, split = "val", download = TRUE),
    label = "Dataset should load successfully"
  )
  dl <- torch::dataloader(ds_all, batch_size = 10)
  # 5400 turns into 540 batches of 10
  expect_length(dl, 540)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  # Check shape, dtype, and values on X
  expect_tensor_shape(i[[1]], c(10, 13, 64, 64))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_tensor_shape(i[[2]], 10)
  expect_tensor_dtype(i[[2]], torch_long())
  expect_named(i, c("x", "y"))})


