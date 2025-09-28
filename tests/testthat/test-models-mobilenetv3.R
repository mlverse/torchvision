context("models-mobilenetv3")

test_that("tests for non-pretrained model_mobilenet_v3_large", {
  model_large <- model_mobilenet_v3_large(pretrained = FALSE)
  input <- torch_randn(1, 3, 224, 224)
  model_large$eval()
  out <- model_large(input)
  expect_tensor_shape(out, c(1000))

  model <- model_mobilenet_v3_large(pretrained = FALSE,num_classes = 10)
  input <- torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(10))

  rm(model_large)
  rm(model)
  gc()
})

test_that("tests for pretrained model_mobilenet_v3_large", {
  model_large <- model_mobilenet_v3_large(pretrained = TRUE)
  input <- torch_randn(1, 3, 224, 224)
  model_large$eval()
  out <- model_large(input)
  expect_tensor_shape(out, c(1000))

  rm(model_large)
  gc()
})

test_that("tests for non-pretrained model_mobilenet_v3_small", {
  model_small <- model_mobilenet_v3_small(pretrained = FALSE)
  input <- torch_randn(1, 3, 224, 224)
  model_small$eval()
  out <- model_small(input)
  expect_tensor_shape(out, c(1000))

  rm(model_small)
  gc()
})

test_that("tests for pretrained model_mobilenet_v3_small", {
  model_small <- model_mobilenet_v3_small(pretrained = TRUE)
  input <- torch_randn(1, 3, 224, 224)
  model_small$eval()
  out <- model_small(input)
  expect_tensor_shape(out, c(1000))

  model <- model_mobilenet_v3_small(pretrained = FALSE,num_classes = 10)
  input <- torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(10))

  rm(model_small)
  rm(model)
  gc()
})

test_that("tests for model_mobilenet_v3_large with non-divisible input shapes", {

  model <- model_mobilenet_v3_large(pretrained = FALSE)
  input <- torch_randn(2, 3, 223, 225)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(2, 1000))
  rm(model)
  gc()
})

test_that("tests for model_mobilenet_v3_small with non-divisible input shapes", {

  model <- model_mobilenet_v3_small(pretrained = FALSE)
  input <- torch_randn(1, 3, 223, 225)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  rm(model)
  gc()
})

test_that("tests for model_mobilenet_v3_small with varied width_mult", {

  model <- model_mobilenet_v3_small(pretrained = FALSE, width_mult = 0.5)
  input <- torch_randn(2, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(2, 1000))
  rm(model)
  gc()
})

test_that("tests for model_mobilenet_v3_large with varied width_mult", {

  model <- model_mobilenet_v3_large(pretrained = FALSE, width_mult = 0.5)
  input <- torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  rm(model)
  gc()
})
