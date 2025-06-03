test_that("fashion_mnist_dataset loads correctly", {
  ds <- fashion_mnist_dataset(
    root = tempdir(),
    train = TRUE,
    download = TRUE
  )

  expect_s3_class(ds, "fashion_mnist_dataset")
  expect_type(ds$.getitem(1), "list")
  expect_named(ds$.getitem(1), c("x", "y"))
  expect_equal(dim(as.array(ds$.getitem(1)$x)), c(28, 28))
  expect_true(ds$.getitem(1)$y >= 1 && ds$.getitem(1)$y <= 10)
})
