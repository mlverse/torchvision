context("dataset-eurosat")


test_that("whoi_small_plankton_dataset downloads correctly whatever the split", {
  skip_on_cran()
  skip_if_not_installed("torch")

  expect_error(
    whoi_small_plankton_dataset(split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    train_ds <- whoi_small_plankton_dataset(split = "train", download = TRUE)
  )

  expect_is(train_ds, "dataset", "train should be a dataset")
  # Train dataset should have exactly 40599 samples
  expect_equal(train_ds$.length(), 40599)

  val_ds <- whoi_small_plankton_dataset(split = "val", download = TRUE, transform = transform_to_tensor)
  # Validation dataset should have exactly 5799 samples
  expect_equal(val_ds$.length(), 5799)

  first_item <- val_ds[1]
  expect_tensor_shape(first_item$x, c(1,145, 230))
  # classification of the first item is "48: Leptocylindrus"
  expect_equal(first_item$y, 47L)
  expect_equal(val_ds$classes[first_item$y], "Leegaardiella_ovalis")

  test_ds <- whoi_small_plankton_dataset(split = "test", download = TRUE)
  # Test dataset should have exactly 11601 samples
  expect_equal(test_ds$.length(), 11601)

})

test_that("whoi_small_plankton_dataset derivatives download and prepare correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")

  expect_error(
    whoi_small_plankton_dataset(split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    val_ds <- whoi_small_plankton_dataset(
      split = "val", download = TRUE,
      transform = . %>% transform_to_tensor() %>% transform_resize(size = c(150, 300))
      )
  )

  dl <- torch::dataloader(val_ds, batch_size = 10)
  # 5799 turns into 580 batches of 10
  expect_length(dl, 580)
  iter <- dataloader_make_iter(dl)
  expect_no_error(
    i <- dataloader_next(iter)
  )

  # Check shape, dtype, and values on X
  expect_tensor_shape(i[[1]], c(10, 1, 150, 300))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_length(i[[2]], 10)
  expect_named(i, c("x", "y"))})




test_that("whoi_plankton_dataset downloads correctly whatever the split", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")


  expect_error(
    whoi_plankton_dataset(split = "test", download = FALSE),
    "Dataset not found. You can use `download = TRUE`",
    label = "Dataset should fail if not previously downloaded"
  )

  expect_no_error(
    train_ds <- whoi_plankton_dataset(split = "train", download = TRUE)
  )

  expect_is(train_ds, "dataset", "train should be a dataset")
  # Train dataset should have exactly 669806 samples
  expect_equal(train_ds$.length(), 669806)

  val_ds <- whoi_plankton_dataset(split = "val", download = TRUE, transform = transform_to_tensor)
  # Validation dataset should have exactly 95686 samples
  expect_equal(val_ds$.length(), 95686)

  first_item <- val_ds[1]
  expect_tensor_shape(first_item$x, c(1,45, 388))
  # classification of the first item is "48: Leptocylindrus"
  expect_equal(first_item$y, 48L)
  expect_equal(val_ds$classes[first_item$y], "Leptocylindrus")

  test_ds <- whoi_plankton_dataset(split = "test", download = TRUE)
  # Test dataset should have exactly 191375 samples
  expect_equal(test_ds$.length(), 191375)

})

test_that("dataloader from whoi_plankton_dataset gets torch tensors", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- whoi_plankton_dataset(
    split = "val", download = TRUE,
    transform = . %>% transform_to_tensor() %>% transform_resize(size = c(150, 300))
  )
  dl <- torch::dataloader(ds, batch_size = 10)
  # 956,9k turns into 9569 batches of 10
  expect_length(dl, 9569)
  iter <- dataloader_make_iter(dl)
  expect_no_error(
    i <- dataloader_next(iter)
  )
  # Check shape, dtype, and values on X
  expect_tensor_shape(i[[1]], c(10, 1, 150, 300))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_length(i[[2]],10)
  expect_named(i, c("x", "y"))})
