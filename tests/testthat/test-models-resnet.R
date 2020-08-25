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
