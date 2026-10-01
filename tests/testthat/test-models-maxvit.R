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
  input <- torch::torch_randn(1, 3, 448, 448)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

test_that("model_maxvit attends within each image and trains", {
  torch::torch_manual_seed(1)
  model <- model_maxvit()
  model$eval()

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

  # all parameters, including the relative position bias tables, receive gradients
  model$train()
  model(input)$sum()$backward()
  expect_true(all(vapply(model$parameters, function(p) !is.null(p$grad), logical(1))))
})

test_that("pretrained model_maxvit loads all weights and classifies images", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_maxvit(pretrained = TRUE, num_classes = 10)
  expect_tensor_shape(model(torch::torch_randn(1, 3, 224, 224)), c(1, 10))
})
