test_that("convert_image_dtype", {

  x <- torch::torch_rand(1, 2, 2, dtype = torch_float32())
  o <- transform_convert_image_dtype(x, torch_int16())
  y <- transform_convert_image_dtype(o, torch_float32())

  expect_equal(round(as_array(x),1), round(as_array(y),1))

})

test_that("normalize", {

  x <- torch_randn(3, 10, 10)
  o <- transform_normalize(x, 1, 2)

  expect_equal_to_r(o, as_array((x - 1)/2))

})

test_that("resize", {

  x <- torch_randn(3, 10, 10)
  o <- transform_resize(x, c(20, 20))

  expect_tensor_shape(o, c(3, 20, 20))
})

test_that("pad", {

  x <- torch_randn(3, 10, 10)
  o <- transform_pad(x, c(1,2))

  expect_tensor_shape(o, c(3, 14, 12))
})

test_that("crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_crop(x, 1, 1, 2, 2)

  expect_tensor_shape(o, c(3,2,2))
  expect_equal(as_array(x[,1,1]), as_array(o[,1,1]))

})

test_that("center_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_center_crop(x, c(2,2))

  expect_tensor_shape(o, c(3,2,2))

})

test_that("resized_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_resized_crop(x, 1, 1, 2, 2, size = c(6, 6))

  expect_tensor_shape(o, c(3,6,6))

})
