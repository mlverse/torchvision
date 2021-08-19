test_that("mobilenetv2 works", {
  model <- model_mobilenet_v2()
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))

  model <- model_mobilenet_v2(pretrained = TRUE)
  torch::torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 256, 256)
  out <- model(input)

  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1,1], -1.1959798336029053, tolerance = 1e-6) # value taken from pytorch
})
