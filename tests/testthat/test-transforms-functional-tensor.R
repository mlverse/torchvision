test_that("crop", {

  img <- torch_randn(3, 32, 32)
  o <- tft_crop(img, top = 5, left = 5, width = 5, height = 5)

  expect_equal(o$shape, c(3, 5, 5))
})

test_that("hflip", {

  img <- torch_randn(3, 32, 32)

  expect_true(
    torch::torch_allclose(img, tft_hflip(tft_hflip(img)))
  )
})

test_that("vflip", {

  img <- torch_randn(3, 32, 32)

  expect_true(
    torch::torch_allclose(img, tft_vflip(tft_vflip(img)))
  )
})

test_that("rgb_to_grayscale", {

  img <- torch_randn(3, 32, 32)
  im <- tft_rgb_to_grayscale(img)

  expect_equal(dim(im), c(32, 32))

})

test_that("adjust_brightness", {

  img <- torch_randn(3, 32, 32)
  im <- tft_adjust_brightness(img, 0.5)

  expect_tensor_shape(im, c(3,32,32))

})

test_that("adjust_contrast", {

  img <- torch_randn(3, 32, 32)
  im <- tft_adjust_contrast(img, 0.5)

  expect_tensor_shape(im, c(3,32,32))
})

test_that("adjust_hue", {

  img <- torch_randn(3, 32, 32)
  im <- tft_adjust_hue(img, 0.5)

  expect_tensor_shape(im, c(3,32,32))
})

test_that("adjust_saturation", {

  img <- torch_randn(3, 32, 32)
  im <- tft_adjust_saturation(img, 0.5)

  expect_tensor_shape(im, c(3,32,32))
})

test_that("adjust_gamma", {

  img <- torch_randn(3, 32, 32)
  im <- tft_adjust_gamma(img, 0.5)

  expect_tensor_shape(im, c(3,32,32))
})

test_that("center_crop", {

  img <- torch_randn(3, 32, 32)
  im <- tft_center_crop(img, c(25, 25))

  expect_tensor_shape(im, c(3, 25, 25))

})

test_that("five_crop", {

  img <- torch_randn(3, 32, 32)
  r <- tft_five_crop(img, size = c(10, 10))

  lapply(r, function(x) expect_tensor_shape(x, c(3, 10, 10)))

})

test_that("ten_crop", {

  img <- torch_randn(3, 32, 32)
  r <- tft_ten_crop(img, size = c(10, 10))

  lapply(r, function(x) expect_tensor_shape(x, c(3, 10, 10)))

})

test_that("pad", {

  img <- torch_randn(3, 32, 32)
  im <- tft_pad(img, c(2))

  expect_tensor_shape(im, c(3, 36, 36))
  expect_equal_to_r(im[,1,1], c(0,0,0))

})

test_that("resize", {

  img <- torch_rand(3, 32, 32)
  im <- tft_resize(img, c(40, 40))

  expect_tensor_shape(im, c(3, 40, 40))
  expect_true(!any(torch::as_array(torch_isnan(im)))) # no nans

})

