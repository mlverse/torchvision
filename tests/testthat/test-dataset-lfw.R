context("dataset-lfw")

t <- withr::local_tempdir()

test_that("tests for the LFW People dataset for original image_set", {
  # MacOS / Windows runner fails with `cannot open URL 'https://ndownloader.figshare.com/files/5976015'` or `timeout`
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_people_dataset(root = t, download = TRUE, split = "original")
  expect_length(lfw, 13233)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 187500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW People dataset for funneled image_set", {
  # MacOS / Windows runner fails with `cannot open URL 'https://ndownloader.figshare.com/files/5976015'` or `timeout`
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_people_dataset(root = t, download = TRUE, split = "funneled" )
  expect_length(lfw, 13233)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 187500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW People dataset for dataloader", {
  # MacOS / Windows runner fails with `cannot open URL 'https://ndownloader.figshare.com/files/5976015'` or `timeout`
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
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
})

test_that("tests for the LFW Pairs dataset for original image_set train split", {
  # MacOS runner fails with cannot open URL 'https://ndownloader.figshare.com/files/5976015'
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_pairs_dataset(root = t, download = TRUE, split = "original", train = TRUE)
  expect_length(lfw, 2200)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 2)
  expect_length(first_item$x[[1]], 187500)
  expect_length(first_item$x[[2]], 187500)
  expect_type(first_item$x, "list")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW Pairs dataset for funneled image_set train split", {
  # MacOS runner fails with cannot open URL 'https://ndownloader.figshare.com/files/5976015'
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_pairs_dataset(root = t, train = TRUE, split = "funneled", download = TRUE)
  expect_length(lfw, 2200)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 2)
  expect_length(first_item$x[[1]], 187500)
  expect_length(first_item$x[[2]], 187500)
  expect_type(first_item$x, "list")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW Pairs dataset for original image_set test split", {
  # MacOS runner fails with cannot open URL 'https://ndownloader.figshare.com/files/5976015'
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_pairs_dataset(root = t, split = "original", train = FALSE)
  expect_length(lfw, 1000)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 2)
  expect_length(first_item$x[[1]], 187500)
  expect_length(first_item$x[[2]], 187500)
  expect_type(first_item$x, "list")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW Pairs dataset for funneled image_set test split", {
  # MacOS runner fails with cannot open URL 'https://ndownloader.figshare.com/files/5976015'
  skip_on_os("mac")
  skip_on_os("windows")
  skip_on_os("linux")
  lfw <- lfw_pairs_dataset(root = t, train = FALSE, split = "funneled")
  expect_length(lfw, 1000)
  first_item <- lfw[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x, 2)
  expect_length(first_item$x[[1]], 187500)
  expect_length(first_item$x[[2]], 187500)
  expect_type(first_item$x, "list")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the LFW People dataset for dataloader", {
  # MacOS runner fails with cannot open URL 'https://ndownloader.figshare.com/files/5976015'
  skip_on_os("mac")
  skip_on_os("windows")
  lfw <- lfw_pairs_dataset(
    root = t,
    transform = function(pair) {
      pair[[1]] %>% transform_to_tensor()
      pair[[2]] %>% transform_to_tensor()
      pair
    },
    download = TRUE)
  dl <- dataloader(lfw, batch_size = 32)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_named(batch, c("x", "y"))
  expect_length(batch$x, 2)
  expect_type(batch$x, "list")
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y, 32)
  expect_tensor_dtype(batch$y, torch_long())
  expect_equal_to_r(batch$y[1], 1)
  expect_equal_to_r(batch$y[2], 1)
  expect_equal_to_r(batch$y[32], 1)
})
