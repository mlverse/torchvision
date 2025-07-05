test_that("efficientnet v2 models produce correct output shapes", {

  variants <- list(
    s = model_efficientnet_v2_s,
    m = model_efficientnet_v2_m,
    l = model_efficientnet_v2_l
  )

  for (name in names(variants)) {
    fn <- variants[[name]]

    # without pretrained
    model <- fn(pretrained = FALSE)
    input <- torch::torch_randn(1, 3, 256, 256)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))

    # with pretrained
    withr::with_options(list(timeout = 360), {
      model <- fn(pretrained = TRUE)
    })
    input <- torch::torch_randn(1, 3, 256, 256)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))
  }
})
