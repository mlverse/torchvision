test_that("random_resised_crop", {

  img <- torch::torch_randn(3, 224, 224)
  o <- transform_random_resized_crop(img, size = c(32, 32))
  expect_tensor_shape(o, c(3, 32,32))

  im <- magick::image_read("torch.png")
  o <- transform_random_resized_crop(im, size = c(32, 32))
  expect_tensor_shape(transform_to_tensor(o), c(3, 32, 32))

})
