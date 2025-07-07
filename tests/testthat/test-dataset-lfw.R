context("dataset-lfw")

t <- withr::local_tempdir()

test_that("tests for the LFW People dataset for train split", {

  lfw <- lfw_people_dataset(root = t, download = TRUE)
  expect_length(lfw, 9525)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 187500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW People dataset for test split", {

  lfw <- lfw_people_dataset(root = t, train = FALSE)
  expect_length(lfw, 3708)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 187500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW People dataset for dataloader", {

  lfw <- lfw_people_dataset(root = t, transform = transform_to_tensor)
  dl <- dataloader(lfw, batch_size = 32)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_named(batch, c("x", "y"))
  expect_length(batch$x, 6000000)
  expect_tensor(batch$x)
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor_dtype(batch$y,torch_long())
  expect_tensor_shape(batch$x,c(32,3,250,250))
  expect_tensor_shape(batch$y,32)
  expect_tensor(batch$y)
  expect_equal_to_r(batch$y[1], 1)
  expect_equal_to_r(batch$y[2], 2)
  expect_equal_to_r(batch$y[32], 17)
})