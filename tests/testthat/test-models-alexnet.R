test_that("alexnet", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

    m <- model_alexnet()
  input <- torch::torch_randn(1, 3, 256, 256)

  out <- m(input)

  expect_tensor_shape(out, c(1, 1000))

  m <- model_alexnet(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 256, 256)

  out <- m(input)

  expect_tensor_shape(out, c(1, 1000))

})
