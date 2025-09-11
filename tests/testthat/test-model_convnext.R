context("models-convnext")

test_that("non-pretrained model_convnext_tiny work", {
  expect_no_error(
    model_tiny <- model_convnext_tiny(pretrained = FALSE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_tiny$eval()
  out <- model_tiny(input)
  expect_tensor_shape(out, c(1, 1000))

  expect_no_error(
    model <- model_convnext_tiny(pretrained = FALSE, num_classes = 10)
  )
  input <- torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model_tiny)
  rm(model)
  gc()
})

test_that("pretrained model_convnext_tiny works", {
  expect_no_error(
    model_tiny <- model_convnext_tiny(pretrained = TRUE)
  )
  input <- torch_randn(1, 3, 224, 224)
  model_tiny$eval()
  out <- model_tiny(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model_tiny)
  gc()
})
