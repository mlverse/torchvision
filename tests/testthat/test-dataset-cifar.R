context("dataset-cifar")

t <- withr::local_tempdir()

test_that("cifar10", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  expect_error(
    ds <- cifar10_dataset(root = tempfile(), train = TRUE),
    class = "runtime_error"
  )

  withr::with_options(list(timeout = 3600), {
    ds <- cifar10_dataset(root = t, train = TRUE, download = TRUE)
  })
  expect_length(ds, 50000)
  el <- ds[1]
  expect_equal(dim(el[[1]]), c(32, 32, 3))
  expect_length(el[[2]], 1)
  expect_named(el, c("x", "y"))
  expect_equal(ds$classes[el[[2]]], "frog")

  withr::with_options(list(timeout = 3600), {
    ds <- cifar10_dataset(root = t, train = FALSE, download = TRUE)
  })
  expect_length(ds, 10000)
  el <- ds[1]
  expect_equal(dim(el[[1]]), c(32, 32, 3))
  expect_length(el[[2]], 1)
  expect_named(el, c("x", "y"))

})

test_that("cifar100", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  expect_error(
    ds <- cifar100_dataset(root = tempfile(), train = TRUE),
    class = "runtime_error"
  )

  withr::with_options(list(timeout = 3600), {
    ds <- cifar100_dataset(root = t, train = TRUE, download = TRUE)
  })
  expect_length(ds, 50000)
  el <- ds[2500]
  expect_equal(dim(el[[1]]), c(32, 32, 3))
  expect_length(el[[2]], 1)
  expect_named(el, c("x", "y"))
  expect_equal(ds$classes[el[[2]]], "motorcycle")

  withr::with_options(list(timeout = 3600), {
    ds <- cifar100_dataset(root = t, train = FALSE, download = TRUE)
  })
  expect_length(ds, 10000)
  el <- ds[502]
  expect_equal(dim(el[[1]]), c(32, 32, 3))
  expect_length(el[[2]], 1)
  expect_named(el, c("x", "y"))
  expect_equal(ds$classes[el[[2]]], "mouse")

})
