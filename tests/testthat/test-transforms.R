test_that("convert_image_dtype", {

  x <- torch::torch_rand(1, 2, 2, dtype = torch_float32())
  o <- transform_convert_image_dtype(x, torch_int16())
  y <- transform_convert_image_dtype(o, torch_float32())

  expect_identical(round(as_array(x),1), round(as_array(y),1))

})

test_that("normalize", {

  x <- torch_randn(3, 10, 10)
  o <- transform_normalize(x, 1, 2)

  expect_equal_to_r(o, as_array((x - 1)/2))

})

test_that("normalize error is glued", {

  x <- torch_randn(3, 10, 10)

  expect_error(transform_normalize(x, 1, 0), "evaluated to zero after conversion to Float")

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
  expect_identical(as_array(x[,1,1]), as_array(o[,1,1]))

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

  expect_tensor_shape(output, c(1,4,4))
  expect_equal_to_r(output[1,,1], c(4,3,2,1))

  output <- transform_rotate(img, 45, expand = TRUE)
  expect_equal_to_r(output[1,,2], c(0,0, 2, 5, 0, 0))
  expect_equal_to_r(output[1,,3], c(0,3, 7, 10, 9, 0))

})

test_that("rotate a rectangle image", {

  img <- torch::torch_tensor(matrix(1:20))$view(c(1, 4, 5))
  output <- transform_rotate(img, 90)

  expect_tensor_shape(output, c(1,5,4))
  expect_equal_to_r(output[1,,1], c(5,4,3,2,1))
  expect_equal_to_r(output[1,,4], c(20,19,18,17,16))

})

test_that("random_affine", {

  x <- torch_eye(8)$view(c(1, 1, 8, 8))

  # no translation
  o <- transform_random_affine(x, 0, c(0, 0))
  expect_identical(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

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
  expect_identical(as.numeric(torch_sum(x)) - 1, as.numeric(torch_sum(o)))

  # translate by 1 pixel vertically
  # should result in sum smaller by 1
  o <- transform_affine(x, 0, c(1, 0), 1, 0)
  expect_identical(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))

})

test_that("linear transformation", {

  c <- 3
  h <- 24
  w <- 32

  tensor <- torch::torch_randn(c, h, w)
  matrix <- torch::torch_rand(c * h * w, c * h * w)
  mean_vector <- torch::torch_rand(c * h * w)

  out <- transform_linear_transformation(tensor, matrix, mean_vector)

  expect_identical(dim(out), c(3, 24, 32))
})

test_that("adjust hue", {

  hue_factor <- c(-0.45, -0.25, 0.0, 0.25, 0.45)
  x <- torch::torch_rand(3, 24, 32)

  for (f in hue_factor) {
    out <- transform_adjust_hue(x, f)
    expect_identical(dim(out), dim(x))
  }

})

test_that("grayscale", {

  x <- torch::torch_rand(3, 24, 32)
  out <- transform_grayscale(x, 3)
  expect_identical(dim(out), dim(x))
  expect_identical(dim(out)[1], 3)

  out <- transform_grayscale(x, 1)
  expect_identical(dim(out)[2:3], dim(x)[2:3])
  expect_identical(dim(out)[1], 1)

})

test_that("random grayscale", {

  tensor <- torch::torch_rand(3, 24, 32)
  for (p in seq(0, 1, length.out = 10)) {
    out <- transform_random_grayscale(tensor, p)
    expect_identical(dim(out), dim(tensor))
  }

})


test_that("random vertical flip", {

  tensor <- torch::torch_randn(3, 24, 32)

  for (i in 1:10) {
    out <- transform_random_vertical_flip(tensor)
    expect_identical(dim(out), dim(tensor))
  }
  for (p in seq(0, 1, length.out = 10)) {
    out <- transform_random_vertical_flip(tensor, p)
    expect_identical(dim(out), dim(tensor))
  }
})


test_that("random rotation works", {

  x <- torch::torch_tensor(array(1, dim = c(3, 200, 200)))

  # Transforms
  rotate <- function(img) transform_random_rotation(img, 20)

  expect_error(rotate(x), regexp = NA)


})

test_that("random choice transform works", {

  # Example Image
  x <- array(1, dim = c(3, 200, 200))

  # Transforms
  color_transform <- function(img) transform_color_jitter(
    img, brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5
  )
  resize_crop <- function(img) transform_random_resized_crop(img, size = c(200, 200))
  hflip <- function(img) transform_random_horizontal_flip(img)
  vflip <- function(img) transform_random_vertical_flip(img)
  rotate <- function(img) transform_random_rotation(img, 20)
  identity <- function(img) img

  # Select a Random Transform to Apply
  expect_error(regexp = NA, {
    transform_random_choice(
      torch_tensor(x),
      list(
        color_transform,
        resize_crop,
        hflip,
        vflip,
        rotate,
        identity
      )
    )
  })

})

