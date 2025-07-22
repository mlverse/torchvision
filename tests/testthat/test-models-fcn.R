test_that("model_fcn_resnet50 returns correct output shape", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet50 works with custom num_classes", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_fcn_resnet50 with aux classifier returns aux output", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet50 loads pretrained weights", {
  model <- model_fcn_resnet50(pretrained = TRUE, num_classes = 21)
  expect_true(inherits(model, "fcn"))
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(2, 21, 224, 224))
  expect_tensor_shape(output$aux, c(2, 21, 224, 224))
})


test_that("model_fcn_resnet101 returns correct output shape", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet101 works with custom num_classes", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_fcn_resnet101 with aux classifier returns aux output", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet101 loads pretrained weights", {
  model <- model_fcn_resnet101(pretrained = TRUE)
  expect_true(inherits(model, "fcn"))
})

