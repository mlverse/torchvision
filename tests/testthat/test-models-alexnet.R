test_that("alexnet", {

  m <- model_alexnet()
  input <- torch::torch_randn(1, 3, 256, 256)

  out <- m(input)

  expect_tensor_shape(out, c(1, 1000))

  m <- model_alexnet(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)

  out <- m(input)

  expect_tensor_shape(out, c(1, 1000))

})
