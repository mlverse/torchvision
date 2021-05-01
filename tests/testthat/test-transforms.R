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

  x <- torch_randn(3, 10, 20)
  o <- transform_resize(x, c(10, 10))
  expect_tensor_shape(o, c(3, 10, 10))

  x <- torch_randn(3, 10, 20)
  o <- transform_resize(x, c(10))
  expect_tensor_shape(o, c(3, 10, 20))

  x <- torch_randn(3, 20, 10)
  o <- transform_resize(x, c(10))
  expect_tensor_shape(o, c(3, 20, 10))

  x <- torch_randn(3, 10, 5)
  o <- transform_resize(x, 10)
  expect_tensor_shape(o, c(3, 20, 10))
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

test_that("hflip", {

  x <- torch_randn(3, 10, 10)
  o <- transform_hflip(x)

  expect_equal_to_r(o[,,1], as_array(x[,,10]))

})

test_that("perspective", {

  skip("not implemented")

  x <- torch_randn(3, 50, 50)
  o <- transform_perspective(x, startpoints = list(c(2,2), c(2,3), c(3,2), c(3,3)),
                             endpoints = list(c(4,4), c(4,5), c(5,4), c(5,5)))

})

test_that("vflip", {

  x <- torch_randn(3, 10, 10)
  o <- transform_vflip(x)

  expect_equal_to_r(o[,1,], as_array(x[,10,]))

})

test_that("five_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_five_crop(x, c(3, 3))

  expect_length(o, 5)

})

test_that("ten_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_ten_crop(x, c(3, 3))

  expect_length(o, 10)

})

test_that("rotate", {

  img <- torch::torch_tensor(matrix(1:16))$view(c(1, 4, 4))
  output <- transform_rotate(img, 90)

  expect_tensor_shape(img, c(1,4,4))
  expect_equal_to_r(output[1,,1], c(4,3,2,1))

  output <- transform_rotate(img, 45, expand = TRUE)
  expect_equal_to_r(output[1,,2], c(0,0, 2, 5, 0, 0))
  expect_equal_to_r(output[1,,3], c(0,3, 7, 10, 9, 0))

})

test_that("random_affine", {

  x <- torch_eye(8)$view(c(1, 1, 8, 8))

  # no translation
  o <- transform_random_affine(x, 0, c(0, 0))
  expect_equal(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

  # probabilistic transformation with p = 0.1 should not result in sum deviating by > 1
  o <- transform_random_affine(x, 0, c(0.1, 0))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

  o <- transform_random_affine(x, 0, c(0, 0.1))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

})

test_that("affine", {

  x <- torch_eye(8)$view(c(1, 1, 8, 8))

  # translate by 1 pixel horizontally
  # should result in sum smaller by 1
  o <- transform_affine(x, 0, c(0, 1), 1, 0)
  expect_equal(as.numeric(torch_sum(x)) - 1, as.numeric(torch_sum(o)))

  # translate by 1 pixel vertically
  # should result in sum smaller by 1
  o <- transform_affine(x, 0, c(1, 0), 1, 0)
  expect_equal(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))

})


