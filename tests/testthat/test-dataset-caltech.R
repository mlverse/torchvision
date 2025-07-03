context("dataset-caltech")

t <- withr::local_tempdir()

test_that("Caltech101 dataset works correctly", {

  expect_error(
    caltech101_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  ds <- caltech101_dataset(root = t, download = TRUE)
  expect_equal(length(ds), 8677)
  first_item <- ds[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("Caltech101 dataset works correctly (dataloader)", {

  caltech101 <- caltech101_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(caltech101, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_dtype(batch$y, torch_long())
  expect_tensor_shape(batch$y, 4)
  expect_tensor(batch$y[1])
  expect_tensor_dtype(batch$y[1],torch_long())
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],1)
  expect_equal_to_r(batch$y[3],1)
  expect_equal_to_r(batch$y[4],1)

})

test_that("Caltech256 dataset works correctly", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS") != "1",
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  expect_error(
    caltech256_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  caltech256 <- caltech256_dataset(root = t, download = TRUE)
  expect_length(caltech256, 30607)
  first_item <- caltech256[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,416166)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("Caltech256 dataset works correctly (dataloader)", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS") != "1",
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  caltech256 <- caltech256_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(caltech256, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_dtype(batch$y, torch_long())
  expect_tensor_shape(batch$y, 4)
  expect_tensor(batch$y[1])
  expect_tensor_dtype(batch$y[1],torch_long())
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],1)
  expect_equal_to_r(batch$y[3],1)
  expect_equal_to_r(batch$y[4],1)
  
})