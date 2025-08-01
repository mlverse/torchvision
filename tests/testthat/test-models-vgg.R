test_that("vgg models works", {

  vggs <- list(
    model_vgg11,
    model_vgg11_bn,
    model_vgg13,
    model_vgg13_bn,
    model_vgg16,
    model_vgg16_bn,
    model_vgg19,
    model_vgg19_bn
  )

  for (m in vggs) {

    model <- m()
    expect_tensor_shape(model(torch_ones(5, 3, 224, 224)), c(5, 1000))

  }

  skip_on_ci() # unfortunatelly we don't have enough RAM on CI for that.
  #skip_on_os(os = "mac") # not downloading a bunch of files locally.
  #skip_on_os(os = "windows") # not downloading a bunch of files locally.

  for (m in vggs) {
    model <- m(pretrained = TRUE)
    expect_tensor_shape(model(torch_ones(1, 3, 224, 224)), c(1, 1000))
    gc()
  }
  unlink_model_file()
})
