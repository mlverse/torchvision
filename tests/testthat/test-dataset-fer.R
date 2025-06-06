test_that("fer_dataset", {

  t <- tempfile()

  expect_error(
    ds <- fer_dataset(root = t, train = TRUE),
    class = "runtime_error"
  )

  ds <- fer_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(ds), dim(ds$data_array)[1])

  el <- ds[1]
  expect_true(is.list(el))
  expect_named(el, c("x", "y"))
  
  actual_dims <- dim(el$x)
  expect_true(length(actual_dims) >= 3)
  expect_equal(actual_dims[2:4], c(48, 48))

  expect_equal(length(el$y), 1)

  ds <- fer_dataset(root = t, train = FALSE, download = TRUE)
  expect_equal(length(ds), dim(ds$data_array)[1])

  el <- ds[1]
  expect_named(el, c("x", "y"))
  
  actual_dims <- dim(el$x)
  expect_true(length(actual_dims) >= 3)
  expect_equal(actual_dims[2:4], c(48, 48))
  expect_equal(length(el$y), 1)

})
