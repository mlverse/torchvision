context("dataset-fashionmnist")

test_that("tests for the FashionMNIST dataset", {

  dir <- tempfile(fileext = "/")

  expect_error(
    ds <- fashion_mnist_dataset(dir)
  )

  ds <- fashion_mnist_dataset(dir, download = TRUE)

  i <- ds[1]
  expect_equal(dim(i[[1]]), c(28, 28))
  expect_equal(i[[2]], 10)  # Example label; adjust based on actual dataset content
  expect_equal(length(ds), 60000)

  ds <- fashion_mnist_dataset(dir, transform = transform_to_tensor)
  dl <- torch::dataloader(ds, batch_size = 32)
  expect_length(dl, 1875)
  iter <- dataloader_make_iter(dl)
  i <- dataloader_next(iter)
  expect_tensor_shape(i[[1]], c(32, 1, 28, 28))
  expect_tensor_shape(i[[2]], 32)
  expect_true((torch_max(i[[1]]) <= 1)$item())
  expect_named(i, c("x", "y"))

})
