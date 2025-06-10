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

test_that("fashion_mnist_dataset loads correctly", {
  dir <- tempfile()

  ds <- fashion_mnist_dataset(
    root = dir,
    train = TRUE,
    download = TRUE
  )

  expect_s3_class(ds, "fashion_mnist_dataset")
  expect_type(ds$.getitem(1), "list")
  expect_named(ds$.getitem(1), c("x", "y"))
  expect_equal(dim(as.array(ds$.getitem(1)$x)), c(28, 28))
  expect_true(ds$.getitem(1)$y >= 1 && ds$.getitem(1)$y <= 10)

  ds2 <- fashion_mnist_dataset(dir, transform = transform_to_tensor)
  dl <- torch::dataloader(ds2, batch_size = 32)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_tensor_shape(batch$x, c(32, 1, 28, 28))
  expect_tensor_shape(batch$y, 32)
  expect_named(batch, c("x", "y"))
})

test_that("tests for the emnist dataset", {
  dir <- tempfile()

  ds <- emnist_dataset(
    root = dir,
    split = "balanced",
    download = TRUE
  )

  expect_s3_class(ds, "emnist_dataset")
  expect_type(ds$.getitem(1), "list")
  expect_named(ds$.getitem(1), c("x", "y"))

  item <- ds$.getitem(1)
  expect_equal(length(dim(item$x)), 2)
  expect_equal(dim(as.array(item$x)), c(28, 28))
  expect_true(item$y >= 1 && item$y <= length(ds$classes))

  ds2 <- emnist_dataset(
    root = dir,
    split = "balanced",
    transform = transform_to_tensor
  )
  dl <- torch::dataloader(ds2, batch_size = 32)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_tensor_shape(batch$x, c(32, 1, 28, 28))
  expect_tensor_shape(batch$y, 32)
  expect_named(batch, c("x", "y"))
})


test_that("tests for the qmnist dataset", {
  dir <- tempfile(fileext = "/")

  expect_error(
      ds <- qmnist_dataset(dir, what = subset)
  )

  splits <- c("train", "test", "nist")

  for (split in splits) {

    ds <- qmnist_dataset(dir, split = split, download = TRUE)

    i <- ds[1]
    expect_equal(dim(i[[1]]), c(28, 28))
    expect_true(i[[2]] %in% 1:10)

    expect_true(length(ds) > 0)

    ds <- qmnist_dataset(dir, split = split, transform = transform_to_tensor)
    dl <- torch::dataloader(ds, batch_size = 32)
    iter <- dataloader_make_iter(dl)
    i <- dataloader_next(iter)
    expect_tensor_shape(i[[1]], c(32, 1, 28, 28))
    expect_tensor_shape(i[[2]], 32)
    expect_true((torch_max(i[[1]]) <= 1)$item())
    expect_named(i, c("x", "y"))
  }
})
