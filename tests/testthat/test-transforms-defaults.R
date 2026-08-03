test_that("random_resized_crop", {

  img <- torch::torch_randn(3, 224, 224)
  o <- transform_random_resized_crop(img, size = c(32, 32))
  expect_tensor_shape(o, c(3, 32,32))

  im <- magick::image_read("torch.png")
  o <- transform_random_resized_crop(im, size = c(32, 32))
  expect_tensor_shape(transform_to_tensor(o), c(3, 32, 32))

})

test_that("center_crop pads a non-square size correctly", {
  x <- torch_ones(3, 16, 20)

  # no padding needed
  expect_equal(dim(transform_center_crop(x, c(8, 10))), c(3, 8, 10))

  # padded on both axes, on one axis only, and with a square size
  expect_equal(dim(transform_center_crop(x, c(24, 30))), c(3, 24, 30))
  expect_equal(dim(transform_center_crop(x, c(24, 10))), c(3, 24, 10))
  expect_equal(dim(transform_center_crop(x, c(8, 30))), c(3, 8, 30))
  expect_equal(dim(transform_center_crop(x, 32)), c(3, 32, 32))

  # the image ends up centred in the padding
  o <- transform_center_crop(torch_ones(1, 2, 2), c(4, 6))
  expect_equal(as.numeric(o$sum()), 4)
  expect_equal_to_r(o[1, 2:3, 3:4], matrix(1, nrow = 2, ncol = 2))
})
