test_that("tests for non-pretrained model_vit_b_16", {
  model <- model_vit_b_16()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_b_16(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model)
  gc()
})

test_that("tests for pretrained model_vit_b_16", {

  model <- model_vit_b_16(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})

test_that("tests for non-pretrained model_vit_b_32", {
  model <- model_vit_b_32()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_b_32(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model)
  gc()
})

test_that("tests for pretrained model_vit_b_32", {

  model <- model_vit_b_32(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})

test_that("tests for non-pretrained model_vit_l_16", {
  model <- model_vit_l_16()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_l_16(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model)
  gc()
})

test_that("tests for pretrained model_vit_l_16", {

  model <- model_vit_l_16(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})

test_that("tests for non-pretrained model_vit_l_32", {
  model <- model_vit_l_32()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  model <- model_vit_l_32(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model)
  gc()
})

test_that("tests for pretrained model_vit_l_32", {

  model <- model_vit_l_32(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})

test_that("tests for model_vit_h_14", {
  model <- model_vit_h_14()
  input <- torch::torch_randn(1, 3, 518, 518)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  skip_if(Sys.info()[["sysname"]] == "Linux", "Skipping on Ubuntu CI")
  model <- model_vit_h_14(num_classes = 10)
  input <- torch::torch_randn(1, 3, 518, 518)
  out <- model(input)
  expect_tensor_shape(out, c(1, 10))

  rm(model)
  gc()
})

test_that("tests for model_vit_h_14", {

  model <- model_vit_h_14(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 518, 518)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})