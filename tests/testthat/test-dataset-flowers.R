context("dataset-flowers")

t <- withr::local_tempdir()

test_that("tests for the Flowers102 dataset for train split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  expect_error(
    flowers102_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  flowers <- flowers102_dataset(root = t, split = "train", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,1131000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for test split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flowers <- flowers102_dataset(root = t, split = "test", download = TRUE)
  expect_length(flowers, 6149)
  first_item <- flowers[1]
  expect_length(first_item$x,784500)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for validation split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flowers <- flowers102_dataset(root = t, split = "val", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,909000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for dataloader", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flowers <- flowers102_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(
    dataset = flowers,
    batch_size = 4
  )
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y,4)
  expect_tensor_dtype(batch$y,torch_long())
  expect_length(batch$y, 4)
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],1)
  expect_equal_to_r(batch$y[3],1)
  expect_equal_to_r(batch$y[4],1)
})
