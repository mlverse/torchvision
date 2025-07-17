test_that("deeplabv3_resnet50 works", {
  model <- model_deeplabv3_resnet50(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet50(pretrained = TRUE)
  })
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet101 works", {
  model <- model_deeplabv3_resnet101(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet101(pretrained = TRUE)
  })
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))
})
