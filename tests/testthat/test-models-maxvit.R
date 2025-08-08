test_that("tests for non-pretrained model_maxvit", {
  model <- model_maxvit()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_maxvit(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))
})

test_that("tests for pretrained model_maxvit", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_maxvit(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})
