context("dataset-flickr")

t <- withr::local_tempdir()

test_that("tests for the flickr8k dataset for train split", {
  skip_on_cran()

  expect_error(
    flickr8k <- flickr8k_caption_dataset(root = tempfile()),
    class = "rlang_error"
  )

  flickr8k <- flickr8k_caption_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr8k, 6000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_length(first_item$x,598500)
  expect_equal(first_item$y,2129)
})

test_that("tests for the flickr8k dataset for test split", {
  skip_on_cran()

  flickr8k <- flickr8k_caption_dataset(root = t, train = FALSE)
  expect_length(flickr8k, 1000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_length(first_item$x,502500)
  expect_equal(first_item$y,5382)
})

test_that("tests for the flickr8k dataset for dataloader", {
  skip_on_cran()
  flickr8k <- flickr8k_caption_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(flickr8k, batch_size = 4)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y, 4)
  expect_tensor_dtype(batch$y, torch_long())
  expect_equal_to_r(batch$y[1],2129)
  expect_equal_to_r(batch$y[2],3332)
  expect_equal_to_r(batch$y[3],5140)
  expect_equal_to_r(batch$y[4],7477)
})

test_that("tests for the flickr30k dataset for train split", {
  skip_on_cran()

  expect_error(
    flickr30k <- flickr30k_caption_dataset(root = tempfile()),
    class = "rlang_error"
  )

  flickr30k <- flickr30k_caption_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr30k, 29000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_length(first_item$x,499500)
  expect_equal(first_item$y,1)
})

test_that("tests for the flickr30k dataset for test split", {
  skip_on_cran()

  flickr30k <- flickr30k_caption_dataset(root = t, train = FALSE)
  expect_length(flickr30k, 1000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_length(first_item$x,691500)
  expect_equal(first_item$y,1)
})

test_that("tests for the flickr30k dataset for dataloader", {
  skip_on_cran()

  flickr30k <- flickr30k_caption_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(flickr30k, batch_size = 4)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y, 4)
  expect_tensor_dtype(batch$y, torch_long())
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],2)
  expect_equal_to_r(batch$y[3],3)
  expect_equal_to_r(batch$y[4],4)
})