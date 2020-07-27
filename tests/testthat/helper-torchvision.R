library(torch)

is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

expect_no_error <- function(object, ...) {
  expect_error(object, NA, ...)
}

expect_tensor_shape <- function(object, expected) {
  expect_tensor(object)
  expect_equal(object$shape, expected)
}

expect_tensor <- function(object) {
  expect_true(is_torch_tensor(object))
  expect_no_error(torch::as_array(object))
}

expect_equal_to_r <- function(object, expected) {
  expect_equal(torch::as_array(object), expected)
}
