test_that("efficientnet v2 small model produce correct output shapes", {

  # without pretrained
  model <- model_efficientnet_v2_s(pretrained = FALSE)
  input <- torch::torch_randn(1, 3, 384, 384)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  # with pretrained
  withr::with_options(list(timeout = 360), {
    model <- model_efficientnet_v2_s(pretrained = TRUE)
  })
  input <- torch::torch_randn(1, 3, 384, 384)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))

  unlink_model_file()
})


test_that("efficientnet v2 models produce correct output shapes", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  variants <- list(
    m = model_efficientnet_v2_m,
    l = model_efficientnet_v2_l
  )

  sizes <- c(m = 480, l = 512)

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
})

test_that("efficientnet v2 allows custom num_classes", {
  model <- model_efficientnet_v2_s(num_classes = 17)
  input <- torch::torch_randn(1, 3, 384, 384)
  out <- model(input)
  expect_tensor_shape(out, c(1, 17))
})
