context("dataset-caltech")

test_that("tests for the Caltech101 dataset", {
  t <- tempfile()

  expect_error(
    caltech101_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  caltech101 <- caltech101_dataset(root = t, download = TRUE)
  expect_equal(length(caltech101), 8677)
  first_item <- caltech101[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), "accordion")

})

test_that("tests for the Caltech256 dataset", {
  t <- tempfile()

  expect_error(
    caltech256_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  caltech256 <- caltech256_dataset(root = t, download = TRUE)
  expect_equal(length(caltech256), 30607)
  first_item <- caltech256[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), "001.ak47")

})