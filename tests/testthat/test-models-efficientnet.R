test_that("efficientnet models work", {
  skip_on_os(c("windows", "mac"))

  models <- list(
    model_efficientnet_b0,
    model_efficientnet_b1,
    model_efficientnet_b2,
    model_efficientnet_b3,
    model_efficientnet_b4,
    model_efficientnet_b5,
    model_efficientnet_b6,
    model_efficientnet_b7
  )

  for (m in models) {
    model <- m()
    input <- torch::torch_randn(1, 3, 256, 256)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))
  }

  for (m in models) {
    withr::with_options(list(timeout = 360), model <- m(pretrained = TRUE))
    input <- torch::torch_randn(1, 3, 256, 256)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))
    rm(model)
    gc()
  }
})
