context("dataset-mnist")

test_that("tests for the mnist dataset", {

  dir <- tempfile(fileext = "/")

  expect_error(
    ds <- mnist_dataset(dir)
  )

  ds <- mnist_dataset(dir, download = TRUE)

  i <- ds[1]
  expect_equal(dim(i[[1]]), c(28, 28))
  expect_equal(i[[2]], 6)
  expect_equal(length(ds), 60000)

  ds <- mnist_dataset(dir, transform = transform_to_tensor)
  dl <- torch::dataloader(ds, batch_size = 32)
  expect_length(dl, 1875)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  expect_tensor_shape(i[[1]], c(32, 1, 28, 28))
  expect_tensor_shape(i[[2]], 32)
  expect_true((torch_max(i[[1]]) <= 1)$item())
  expect_named(i, c("x", "y"))

})


test_that("tests for the kmnist dataset", {

  dir <- tempfile(fileext = "/")

  expect_error(
    ds <- kmnist_dataset(dir)
  )

  ds <- kmnist_dataset(dir, download = TRUE)

  i <- ds[1]
  expect_equal(dim(i[[1]]), c(28, 28))
  expect_equal(i[[2]], 9)
  expect_equal(length(ds), 60000)

  ds <- kmnist_dataset(dir, transform = transform_to_tensor)
  dl <- torch::dataloader(ds, batch_size = 32)
  expect_length(dl, 1875)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  expect_tensor_shape(i[[1]], c(32, 1, 28, 28))
  expect_tensor_shape(i[[2]], 32)
  expect_true((torch_max(i[[1]]) <= 1)$item())
  expect_named(i, c("x", "y"))
})

test_that("tests for the emnist dataset (all splits)", {

  dir <- tempfile(fileext = "/")
  splits <- c("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
  expected_lengths <- c(
    byclass = 697932,
    bymerge = 697932,
    balanced = 112800,
    letters = 88800,
    digits = 240000,
    mnist = 60000
  )
  expected_classes <- c(
    byclass = 62,
    bymerge = 47,
    balanced = 47,
    letters = 26,
    digits = 10,
    mnist = 10
  )

  for (split in splits) {

    ds <- emnist_dataset(dir, split = split, download = TRUE)
    i <- ds[1]

    expect_equal(dim(i[[1]]), c(28, 28))
    expect_true(i[[2]] >= 0 && i[[2]] < expected_classes[[split]])
    expect_equal(length(ds), expected_lengths[[split]])

    ds <- emnist_dataset(dir, split = split, transform = transform_to_tensor)
    dl <- torch::dataloader(ds, batch_size = 32)
    iter <- dataloader_make_iter(dl)
    i <- dataloader_next(iter)

    expect_tensor_shape(i[[1]], c(32, 1, 28, 28))
    expect_tensor_shape(i[[2]], 32)
    expect_true((torch_max(i[[1]]) <= 1)$item())
    expect_named(i, c("x", "y"))
  }

})
