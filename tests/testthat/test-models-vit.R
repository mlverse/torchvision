test_that("tests for model_vit_b_16", {
  model <- model_vit_b_16()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_b_16(pretrained = TRUE)
  torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1, 1], -0.87923414, tol = 1e-8)

  rm(model)
  gc()
})

test_that("tests for model_vit_b_32", {
  model <- model_vit_b_32()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_b_32(pretrained = TRUE)
  torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1, 1], -0.54261702, tol = 1e-8)

  rm(model)
  gc()
})

test_that("tests for model_vit_l_16", {
  model <- model_vit_l_16()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_l_16(pretrained = TRUE)
  torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1, 1], -0.78047544, tol = 1e-8)

  rm(model)
  gc()
})

test_that("tests for model_vit_l_32", {
  model <- model_vit_l_32()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_l_32(pretrained = TRUE)
  torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1, 1], -0.78083247, tol = 1e-8)

  rm(model)
  gc()
})

test_that("tests for model_vit_h_14", {
  model <- model_vit_h_14()
  input <- torch::torch_randn(1, 3, 518, 518)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_h_14(pretrained = TRUE)
  torch_manual_seed(1)
  input <- torch::torch_randn(1, 3, 518, 518)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
  expect_equal_to_r(out[1, 1], -0.98756719, tol = 1e-8)

  rm(model)
  gc()
})