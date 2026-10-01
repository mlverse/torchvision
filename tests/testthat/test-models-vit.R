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

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

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

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

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

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

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

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_vit_h_14(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 518, 518)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  rm(model)
  gc()
})
test_that("model_vit attends over the patches of each image", {
  torch::torch_manual_seed(1)
  model <- model_vit_b_32()
  model$eval()
  expect_true("pos_embedding" %in% names(model$parameters))

  input <- torch::torch_randn(2, 3, 224, 224)
  changed <- input$clone()
  changed[2, , , ] <- torch::torch_randn(3, 224, 224)
  torch::with_no_grad({
    out <- model(input)
    out_changed <- model(changed)
  })
  # the prediction of an image depends on the image itself ...
  expect_gt(as.numeric((out[2, ] - out_changed[2, ])$abs()$max()), 1e-4)
  # ... but not on the other images of the batch
  expect_equal(as.numeric((out[1, ] - out_changed[1, ])$abs()$max()), 0, tolerance = 1e-5)

  rm(model)
  gc()
})
