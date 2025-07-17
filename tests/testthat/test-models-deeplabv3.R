test_that("deeplabv3_resnet50 works with default aux_loss=NULL", {
  model <- model_deeplabv3_resnet50(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet50(pretrained = TRUE)
  })
  out <- model(input)
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet101 works with default aux_loss=NULL", {
  model <- model_deeplabv3_resnet101(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet101(pretrained = TRUE)
  })
  out <- model(input)
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet50 works with aux_loss = TRUE", {
  model <- model_deeplabv3_resnet50(aux_loss = TRUE, num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet50 works with aux_loss = FALSE", {
  model <- model_deeplabv3_resnet50(aux_loss = FALSE, num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("custom num_classes works with aux_loss = TRUE", {
  model <- model_deeplabv3_resnet50(num_classes = 3, aux_loss = TRUE)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 3, 32, 32))
  expect_tensor_shape(out$aux, c(1, 3, 32, 32))
})

test_that("pretrained requires num_classes = 21", {
  expect_error(
    model_deeplabv3_resnet50(pretrained = TRUE, num_classes = 3),
    "num_classes = 21"
  )
})
