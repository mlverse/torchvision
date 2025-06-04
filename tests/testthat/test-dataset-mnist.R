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

test_that("tests for the emnist dataset", {

  dir <- tempfile(fileext = "/")

  splits <- list(
    byclass  = list(n_classes = 62, n_train = 697932),
    bymerge  = list(n_classes = 47, n_train = 697932),
    balanced = list(n_classes = 47, n_train = 112800),
    letters  = list(n_classes = 26, n_train = 124800),
    digits   = list(n_classes = 10, n_train = 240000),
    mnist    = list(n_classes = 10, n_train = 60000)
  )

  for (split in names(splits)) {
    info <- splits[[split]]
    #cat("Testing split:", split, "\n")

    expect_error(
      ds <- emnist_dataset(dir, split = split)
    )

    ds <- emnist_dataset(dir, split = split, download = TRUE)
    
    i <- ds[1]
    expect_equal(dim(i[[1]]), c(28, 28))
    expect_true(i[[2]] %in% 0:(info$n_classes - 1))
    expect_equal(length(ds), info$n_train)

    ds <- emnist_dataset(dir, split = split, transform = transform_to_tensor)
    dl <- dataloader(ds, batch_size = 32)
    expect_length(dl, ceiling(info$n_train / 32))

    iter <- dataloader_make_iter(dl)
    batch <- dataloader_next(iter)

    expect_tensor_shape(batch[[1]], c(32, 1, 28, 28))
    expect_tensor_shape(batch[[2]], 32)
    expect_true((torch_max(batch[[1]]) <= 1)$item())
    expect_named(batch, c("x", "y"))
  }
})
