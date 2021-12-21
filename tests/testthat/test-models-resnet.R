test_that("resnet18", {

  model <- model_resnet18()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  model <- model_resnet18(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnet34", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnet34()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  model <- model_resnet34(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnet50", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnet50()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  model <- model_resnet50(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnet101", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnet101()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_resnet101(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnet152", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnet152()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_resnet152(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnext50_32x4d", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnext50_32x4d()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_resnext50_32x4d(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnext50_32x4d", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnext50_32x4d()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_resnext50_32x4d(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("resnext101_32x8d", {
  skip_on_os(c("windows", "mac"))

  model <- model_resnext101_32x8d()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_resnext101_32x8d(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("wide_resnet50_2", {
  skip_on_os(c("windows", "mac"))

  model <- model_wide_resnet50_2()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_wide_resnet50_2(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})

test_that("wide_resnet101_2", {
  skip_on_os(c("windows", "mac"))

  model <- model_wide_resnet101_2()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  withr::with_options(list(timeout = 360),
                      model <- model_wide_resnet101_2(pretrained = TRUE))
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

})
