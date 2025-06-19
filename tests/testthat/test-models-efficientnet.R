test_that("efficientnet_b0", {
  model <- model_efficientnet_b0()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_efficientnet_b0(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b1", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b1()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b1(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b2", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b2()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b2(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b3", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b3()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b3(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b4", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b4()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b4(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b5", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b5()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b5(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b6", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b6()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b6(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("efficientnet_b7", {
  skip_on_os(c("windows", "mac"))

  model <- model_efficientnet_b7()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_efficientnet_b7(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})
