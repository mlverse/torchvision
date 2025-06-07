context("dataset-fer")

test_that("tests for the FER-2013 dataset", {

  t <- tempfile()

  expect_error(
    ds <- fer_dataset(root = t, train = TRUE),
    class = "runtime_error"
  )

  ds <- fer_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(ds), 28709)
  first_item <- ds[1]
  expect_equal(dim(first_item[[1]]), c(1, 48, 48))
  expect_named(first_item, c("x", "y", "class_name"))
  expect_equal(first_item[[2]], 1)
  expect_equal(first_item[[3]], "Angry")

  ds <- fer_dataset(root = t, train = FALSE, download = TRUE)
  first_item <- ds[1]
  expect_equal(dim(first_item[[1]]), c(1, 48, 48))
  expect_equal(length(first_item[[2]]), 1)
  expect_named(first_item, c("x", "y", "class_name"))

})
