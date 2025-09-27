context("dataset-mnist")

dir <- withr::local_tempdir()

test_that("tests for the mnist dataset", {

  expect_error(
    ds <- mnist_dataset(tempfile())
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
  expect_tensor_shape(i[[1]], c(32, 28, 28))
  expect_tensor_shape(i[[2]], 32)
  expect_true((torch_max(i[[1]]) <= 1)$item())
  expect_named(i, c("x", "y"))

})

test_that("tests for the kmnist dataset", {

  expect_error(
    ds <- kmnist_dataset(tempfile())
  )

  ds <- kmnist_dataset(dir, download = TRUE)

  i <- ds[1]
  expect_equal(dim(i[[1]]), c(28, 28))
  expect_equal(i[[2]], 6)
  expect_equal(length(ds), 60000)

  ds <- kmnist_dataset(dir, transform = transform_to_tensor)
  dl <- torch::dataloader(ds, batch_size = 32)
  expect_length(dl, 1875)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  expect_tensor_shape(i[[1]], c(32, 28, 28))
  expect_tensor_shape(i[[2]], 32)
  expect_true((torch_max(i[[1]]) <= 1)$item())
  expect_named(i, c("x", "y"))
})

test_that("fashion_mnist_dataset loads correctly", {

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
  expect_tensor_shape(batch$x, c(32, 28, 28))
  expect_tensor_shape(batch$y, 32)
  expect_named(batch, c("x", "y"))
})

test_that("tests for the emnist dataset", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  expect_error(
    ds <- emnist_collection(root = tempfile())
  )

  emnist <- emnist_collection(dir, dataset = "balanced", download = TRUE)
  expect_equal(length(emnist), 18800)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal((first_item[[2]]), 42)

  emnist <- emnist_collection(dir, dataset = "byclass", split = "test", download = TRUE)
  expect_equal(length(emnist), 116323)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal(dim(first_item$x), c(28,28))
  expect_equal((first_item[[2]]), 19)

  emnist <- emnist_collection(dir, dataset = "bymerge", download = TRUE)
  expect_equal(length(emnist), 116323)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal((first_item[[2]]), 25)

  emnist <- emnist_collection(dir, dataset = "letters", split = "train", download = TRUE,
                           transform = transform_to_tensor)
  expect_equal(length(emnist), 124800)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_tensor(first_item$x)
  expect_tensor_shape(first_item$x, c(1,28,28))
  expect_equal((first_item[[2]]), 24)

  emnist <- emnist_collection(dir, dataset = "digits", download = TRUE)
  expect_equal(length(emnist), 40000)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal((first_item[[2]]), 1)

  emnist <- emnist_collection(dir, dataset = "mnist", split = "train", download = TRUE)
  expect_equal(length(emnist), 60000)
  first_item <- emnist[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal((first_item[[2]]), 5)

  ds2 <- emnist_collection(
    root = dir,
    dataset = "balanced",
    split = "test",
    transform = transform_to_tensor,
    download = TRUE
  )
  dl <- torch::dataloader(ds2, batch_size = 32)
  iter <- torch::dataloader_make_iter(dl)
  batch <- torch::dataloader_next(iter)
  expect_tensor_shape(batch$x, c(32, 1, 28, 28))
  expect_tensor_shape(batch$y, 32)
  expect_named(batch, c("x", "y"))
})


test_that("tests for the qmnist dataset", {

  expect_error(
      ds <- qmnist_dataset(tempfile()),
      "Dataset not found."
  )

  for (split in c("train", "test", "nist")) {

    ds <- qmnist_dataset(dir, split = split, download = TRUE)

    i <- ds[1]
    expect_equal(dim(i[[1]]), c(28, 28))
    expect_true(i[[2]] %in% 1:10)

    expect_true(length(ds) > 0)

    ds <- qmnist_dataset(dir, split = split, transform = transform_to_tensor)
    dl <- torch::dataloader(ds, batch_size = 32)
    iter <- dataloader_make_iter(dl)
    i <- dataloader_next(iter)
    expect_tensor_shape(i[[1]], c(32, 28, 28))
    expect_tensor_shape(i[[2]], 32)
    expect_true((torch_max(i[[1]]) <= 1)$item())
    expect_named(i, c("x", "y"))
  }
})

test_that("tests for the emnist_dataset is deprecated", {

  expect_warning(
    emnist_dataset(kind = "digits"),
    "'emnist_dataset' is deprecated."
  )
})
