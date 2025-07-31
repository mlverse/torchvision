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

expect_tensor_dtype <- function(object, expected_dtype) {
  expect_tensor(object)
  expect_true(object$dtype == expected_dtype)
}

expect_tensor <- function(object) {
  expect_true(is_torch_tensor(object))
  expect_no_error(torch::as_array(object))
}

expect_equal_to_r <- function(object, expected, ...) {
  expect_equal(torch::as_array(object), expected, ...)
}

unlink_model_file <- function() {
  cache_path <- rappdirs::user_cache_dir("torch")
  model_file <- list.files(cache_path, pattern = "*.pth", full.names = TRUE)
  unlink(model_file)
}
