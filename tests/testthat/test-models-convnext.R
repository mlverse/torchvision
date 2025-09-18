context("models-convnext")

test_that("non-pretrained model_convnext_*_1k work, with or wo a changed classification layer", {
  expect_no_error(
    model_1k <- model_convnext_tiny_1k(pretrained = FALSE)
  )
  input <- torch_randn(3, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(3, 1000))

  expect_no_error(
    model_1k <- model_convnext_base_1k(pretrained = FALSE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(1, 1000))

  expect_no_error(
    model_1k <- model_convnext_large_1k(pretrained = FALSE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(1, 1000))

  expect_no_error(
    model <- model_convnext_tiny_1k(pretrained = FALSE, num_classes = 130)
  )
  input <- torch_randn(4, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(4, 130))

  expect_no_error(
    model_1k <- model_convnext_small_22k1k(pretrained = FALSE, num_classes = 37)
  )
  input <- torch_randn(2, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(2, 37))

  expect_no_error(
    model_1k <- model_convnext_base_1k(pretrained = FALSE, num_classes = 10)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(1, 10))

  expect_no_error(
    model_1k <- model_convnext_large_1k(pretrained = FALSE, num_classes = 10)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model_1k)
  rm(model)
  gc()
})

test_that("pretrained model_convnext_*_1k works", {
  expect_no_error(
    model_1k <- model_convnext_tiny_1k(pretrained = TRUE)
  )
  input <- torch_randn(5, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(5, 1000))

  expect_no_error(
    model_1k <- model_convnext_base_1k(pretrained = TRUE)
  )
  input <- torch_randn(4, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(4, 1000))

  rm(model_1k)
  gc()
})

test_that("pretrained model_convnext_*_22k works", {
  expect_no_error(
    model_22k <- model_convnext_tiny_22k(pretrained = TRUE)
  )
  input <- torch_randn(5, 3, 224, 224)
  model_22k$eval()
  out <- model_22k(input)
  expect_tensor_shape(out, c(5, 21841))

  expect_no_error(
    model_22k <- model_convnext_small_22k(pretrained = TRUE)
  )
  input <- torch_randn(2, 3, 224, 224)
  model_22k$eval()
  out <- model_22k(input)
  expect_tensor_shape(out, c(2, 21841))

  expect_no_error(
    model_1k <- model_convnext_small_22k1k(pretrained = FALSE)
  )
  input <- torch_randn(2, 3, 224, 224)
  model_1k$eval()
  out <- model_1k(input)
  expect_tensor_shape(out, c(2, 21841))

  expect_no_error(
    model_22k <- model_convnext_base_22k(pretrained = TRUE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_22k$eval()
  out <- model_22k(input)
  expect_tensor_shape(out, c(1, 21841))

  rm(model_22k)
  gc()
})

test_that("pretrained model_convnext_large_* works", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  expect_no_error(
    model_22k <- model_convnext_large_1k(pretrained = TRUE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_22k$eval()
  out <- model_22k(input)
  expect_tensor_shape(out, c(1, 1000))

  expect_no_error(
    model_22k <- model_convnext_large_22k(pretrained = TRUE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_22k$eval()
  out <- model_22k(input)
  expect_tensor_shape(out, c(1, 21841))

  rm(model_22k)
  gc()
})
