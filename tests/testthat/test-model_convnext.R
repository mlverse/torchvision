context("models-convnext")

test_that("tests for non-pretrained model_convnext_tiny", {
  model_tiny <- model_convnext_tiny(pretrained = FALSE)
  input <- torch_randn(1, 3, 224, 224)
  model_tiny$eval()
  out <- model_tiny(input)
  expect_tensor_shape(out, c(1000))

  model <- model_mobilenet_v3_tiny(pretrained = FALSE,num_classes = 10)
  input <- torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(10))

  rm(model_tiny)
  rm(model)
  gc()
})

test_that("tests for pretrained model_convnext_tiny", {
  model_tiny <- model_convnext_tiny(pretrained = TRUE)
  input <- torch_randn(1, 3, 224, 224)
  model_tiny$eval()
  out <- model_tiny(input)
  expect_tensor_shape(out, c(1000))

  rm(model_tiny)
  gc()
})
