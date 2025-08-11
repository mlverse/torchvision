test_that("efficientnet models produce correct output shapes", {


  variants <- list(
    b0 = model_efficientnet_b0,
    b1 = model_efficientnet_b1,
    b2 = model_efficientnet_b2,
    b3 = model_efficientnet_b3,
    b4 = model_efficientnet_b4,
    b5 = model_efficientnet_b5,
    b6 = model_efficientnet_b6,
    b7 = model_efficientnet_b7
  )

  sizes <- c(b0 = 224, b1 = 240, b2 = 260, b3 = 300,
             b4 = 380, b5 = 456, b6 = 528, b7 = 600)

  for (name in names(variants)) {
    fn <- variants[[name]]
    size <- sizes[[name]]

    # without pretrained
    model <- fn(pretrained = FALSE)
    input <- torch::torch_randn(1, 3, size, size)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))

    # with pretrained
    withr::with_options(list(timeout = 360), {
      model <- fn(pretrained = TRUE)
    })
    input <- torch::torch_randn(1, 3, size, size)
    out <- model(input)
    expect_tensor_shape(out, c(1, 1000))
  }
  unlink_model_file()
})

test_that("efficientnet allows custom num_classes", {
  model <- model_efficientnet_b0(num_classes = 42)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_tensor_shape(out, c(1, 42))
})
