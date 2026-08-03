# each test runs on single tensor image (dim = 3) and with the `b` suffix on batch tensor images (dim = 4)
test_that("convert_image_dtype", {

  x <- torch::torch_rand(1, 2, 2, dtype = torch_float32())
  o <- transform_convert_image_dtype(x, torch_int16())
  y <- transform_convert_image_dtype(o, torch_float32())

  expect_equal(round(as_array(x),1), round(as_array(y),1))

  ob <- transform_convert_image_dtype(x$unsqueeze(1), torch_int16())
  yb <- transform_convert_image_dtype(ob, torch_float32())

  expect_equal(round(as_array(x$unsqueeze(1)),1), round(as_array(yb),1))

})

test_that("normalize", {

  x <- torch_randn(3, 10, 10)
  o <- transform_normalize(x, 1, 2)

  expect_equal_to_r(o, as_array((x - 1)/2))

  ob <- transform_normalize(x$unsqueeze(1), 1, 2)

  expect_equal_to_r(ob, as_array((x$unsqueeze(1) - 1)/2))

})

test_that("normalize error is glued", {

  x <- torch_randn(3, 10, 10)

  expect_error(transform_normalize(x, 1, 0), "evaluated to zero after conversion to Float")
  expect_error(transform_normalize(x$unsqueeze(1), 1, 0), "evaluated to zero after conversion to Float")

})

test_that("resize", {
  x <- torch_randn(3, 10, 10)
  o <- transform_resize(x, c(20, 20))
  expect_tensor_shape(o, c(3, 20, 20))
  ob <- transform_resize(x$unsqueeze(1), c(20, 20))
  expect_tensor_shape(ob, c(1, 3, 20, 20))

  x <- torch_randn(3, 10, 20)
  o <- transform_resize(x, c(10, 10))
  expect_tensor_shape(o, c(3, 10, 10))
  ob <- transform_resize(x$unsqueeze(1), c(10, 10))
  expect_tensor_shape(ob, c(1, 3, 10, 10))

  x <- torch_randn(3, 10, 20)
  o <- transform_resize(x, c(10))
  expect_tensor_shape(o, c(3, 10, 20))
  ob <- transform_resize(x$unsqueeze(1), c(10))
  expect_tensor_shape(ob, c(1, 3, 10, 20))

  x <- torch_randn(3, 20, 10)
  o <- transform_resize(x, c(10))
  expect_tensor_shape(o, c(3, 20, 10))
  ob <- transform_resize(x$unsqueeze(1), c(10))
  expect_tensor_shape(ob, c(1, 3, 20, 10))

  x <- torch_randn(3, 10, 5)
  o <- transform_resize(x, 10)
  expect_tensor_shape(o, c(3, 20, 10))
  ob <- transform_resize(x$unsqueeze(1), 10)
  expect_tensor_shape(ob, c(1, 3, 20, 10))
})

test_that("pad", {

  x <- torch_randn(3, 10, 10)
  o <- transform_pad(x, c(1,2))
  expect_tensor_shape(o, c(3, 14, 12))

  ob <- transform_pad(x$unsqueeze(1), c(1,2))
  expect_tensor_shape(ob, c(1, 3, 14, 12))
})

test_that("crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_crop(x, 1, 1, 2, 2)

  expect_tensor_shape(o, c(3,2,2))
  expect_equal(as_array(x[,1,1]), as_array(o[,1,1]))

  ob <- transform_crop(x$unsqueeze(1), 1, 1, 2, 2)

  expect_tensor_shape(ob, c(1,3,2,2))
  expect_equal(as_array(x$unsqueeze(1)[,,1,1]), as_array(ob[,,1,1]))

})

test_that("center_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_center_crop(x, c(2,2))

  expect_tensor_shape(o, c(3,2,2))

  ob <- transform_center_crop(x$unsqueeze(1), c(2,2))

  expect_tensor_shape(ob, c(1,3,2,2))

})

test_that("resized_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_resized_crop(x, 1, 1, 2, 2, size = c(6, 6))

  expect_tensor_shape(o, c(3,6,6))

  ob <- transform_resized_crop(x$unsqueeze(1), 1, 1, 2, 2, size = c(6, 6))

  expect_tensor_shape(ob, c(1,3,6,6))

})

test_that("hflip", {

  x <- torch_randn(3, 10, 10)
  o <- transform_hflip(x)

  expect_equal_to_r(o[,,1], as_array(x[,,10]))

  ob <- transform_hflip(x$unsqueeze(1))

  expect_equal_to_r(ob[..,1], as_array(x$unsqueeze(1)[..,10]))

})

test_that("perspective", {

  skip("not implemented")

  x <- torch_randn(3, 50, 50)
  o <- transform_perspective(x, startpoints = list(c(2,2), c(2,3), c(3,2), c(3,3)),
                             endpoints = list(c(4,4), c(4,5), c(5,4), c(5,5)))

  ob <- transform_perspective(x$unsqueeze(1), startpoints = list(c(2,2), c(2,3), c(3,2), c(3,3)),
                             endpoints = list(c(4,4), c(4,5), c(5,4), c(5,5)))

})

test_that("vflip", {

  x <- torch_randn(3, 10, 10)
  o <- transform_vflip(x)

  expect_equal_to_r(o[,1,], as_array(x[,10,]))

  ob <- transform_vflip(x$unsqueeze(1))

  expect_equal_to_r(ob[,,1,], as_array(x$unsqueeze(1)[,,10,]))

})

test_that("five_crop", {

  x <- torch_randn(3, 10, 12)
  o <- transform_five_crop(x, c(3, 3))

  expect_length(o, 5)
  expect_tensor_shape(o[[1]], c(3,3,3))

  ob <- transform_five_crop(x$unsqueeze(1), c(3, 3))

  expect_length(ob, 5)
  expect_tensor_shape(ob[[1]], c(1,3,3,3))

})

test_that("ten_crop", {

  x <- torch_randn(3, 10, 10)
  o <- transform_ten_crop(x, c(3, 3))

  expect_length(o, 10)
  expect_tensor_shape(o[[1]], c(3,3,3))

  ob <- transform_ten_crop(x$unsqueeze(1), c(3, 3))

  expect_length(ob, 10)
  expect_tensor_shape(ob[[1]], c(1,3,3,3))
})

test_that("sahi_crop", {

  x <- torch_randn(3, 10, 12)
  sp <- prepare_sahi_split(x, size = c(4, 4), overlap_size_ratio = c(0, 0))
  o <- transform_sahi_crop(x, sp)

  expect_tensor(o)
  expect_tensor_shape(o, c(9,3,4,4))

  h_crops <- ceiling((10 - 4) / 4) + 1
  v_crops <- ceiling((12 - 4) / 4) + 1
  batch_size <- h_crops * v_crops
  expect_tensor_shape(o, c(batch_size, 3, 4, 4))

})

test_that("sahi_crop with non-zero overlap produces correct number of crops", {

  x <- torch_randn(3, 10, 12)
  sp <- prepare_sahi_split(x, size = c(4, 4), overlap_size_ratio = c(0.5, 0.5))
  o <- transform_sahi_crop(x, sp)

  expect_tensor(o)
  expect_equal(length(sp$crop_windows), 20)
  expect_tensor_shape(o, c(20, 3, 4, 4))

})

test_that("sahi_crop returns single stacked crop when size >= image dims", {

  x <- torch_randn(3, 30, 40)
  sp <- prepare_sahi_split(x, size = c(100, 100), overlap_size_ratio = c(0.2, 0.2))
  o <- transform_sahi_crop(x, sp)

  expect_tensor(o)
  expect_tensor_shape(o, c(1,3,30,40))

})

test_that("sahi_crop works with batched 4D tensor input", {

  x <- torch_randn(2, 3, 10, 12)
  sp <- prepare_sahi_split(x, size = c(4, 4), overlap_size_ratio = c(0, 0))
  o <- transform_sahi_crop(x, sp)

  expect_tensor(o)
  expect_equal(o$ndim, 4)
  n_rows <- ceiling((10 - 4) / 4) + 1
  n_cols <- ceiling((12 - 4) / 4) + 1
  expect_tensor_shape(o, c(n_rows * n_cols * 2, 3, 4, 4))

})

test_that("rotate", {

  img <- torch::torch_tensor(matrix(1:16))$view(c(1, 4, 4))
  output <- transform_rotate(img, 90)

  expect_tensor_shape(output, c(1,4,4))
  expect_equal_to_r(output[1,,1], c(4,3,2,1))

  output <- transform_rotate(img, 45, expand = TRUE)
  expect_equal_to_r(output[1,,2], c(0,0, 2, 5, 0, 0))
  expect_equal_to_r(output[1,,3], c(0,3, 7, 10, 9, 0))

  outputb <- transform_rotate(img$unsqueeze(1), 90)

  expect_tensor_shape(outputb, c(1,1,4,4))
  expect_equal_to_r(outputb[,1,,1]$squeeze(1), c(4,3,2,1))

  outputb <- transform_rotate(img$unsqueeze(1), 45, expand = TRUE)
  expect_equal_to_r(outputb[,1,,2]$squeeze(1), c(0,0, 2, 5, 0, 0))
  expect_equal_to_r(outputb[,1,,3]$squeeze(1), c(0,3, 7, 10, 9, 0))

})

test_that("rotate a rectangle image", {

  img <- torch::torch_tensor(matrix(1:20))$view(c(1, 4, 5))
  output <- transform_rotate(img, 90, expand=TRUE)

  expect_tensor_shape(output, c(1,5,4))
  expect_equal_to_r(output[1,,1], c(5,4,3,2,1))
  expect_equal_to_r(output[1,,4], c(20,19,18,17,16))

  outputb <- transform_rotate(img$unsqueeze(1), 90, expand=TRUE)

  expect_tensor_shape(outputb, c(1,1,5,4))
  expect_equal_to_r(outputb[,1,,1]$squeeze(1), c(5,4,3,2,1))
  expect_equal_to_r(outputb[,1,,4]$squeeze(1), c(20,19,18,17,16))

})

test_that("random_affine", {

  x <- torch_eye(8)$view(c(1, 8, 8))

  # no translation
  o <- transform_random_affine(x, 0, c(0, 0))
  expect_equal(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

  ob <- transform_random_affine(x$unsqueeze(1), 0, c(0, 0))
  expect_equal(as.numeric(torch_sum(x)), as.numeric(torch_sum(ob)))

  # probabilistic transformation with p = 0.1 should not result in sum deviating by > 1
  o <- transform_random_affine(x, 0, c(0.1, 0))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

  o <- transform_random_affine(x, 0, c(0, 0.1))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(o)))

  ob <- transform_random_affine(x$unsqueeze(1), 0, c(0.1, 0))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(ob)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(ob)))

  ob <- transform_random_affine(x$unsqueeze(1), 0, c(0, 0.1))
  expect_lte(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(ob)))
  expect_gte(as.numeric(torch_sum(x)), as.numeric(torch_sum(ob)))

})

test_that("affine", {

  x <- torch_eye(8)$view(c(1, 8, 8))

  # translate by 1 pixel horizontally
  # should result in sum smaller by 1
  o <- transform_affine(x, 0, c(0, 1), 1, 0)
  expect_equal(as.numeric(torch_sum(x)) - 1, as.numeric(torch_sum(o)))

  ob <- transform_affine(x$unsqueeze(1), 0, c(0, 1), 1, 0)
  expect_equal(as.numeric(torch_sum(x)) - 1, as.numeric(torch_sum(ob)))

  # translate by 1 pixel vertically
  # should result in sum smaller by 1
  o <- transform_affine(x, 0, c(1, 0), 1, 0)
  expect_equal(as.numeric(torch_sum(x) - 1), as.numeric(torch_sum(o)))

  # interpolation accepts enum-style string values
  o_int <- transform_affine(x, 0, c(0, 1), 1, 0, interpolation = 0)
  o_chr <- transform_affine(x, 0, c(0, 1), 1, 0, interpolation = "nearest")
  expect_equal_to_r(o_chr, as_array(o_int))
})

test_that("affine deprecated arguments still work", {

  x <- torch_eye(8)$view(c(1, 1, 8, 8))

  old_resample <- expect_warning(
    transform_affine(x, 0, c(0, 1), 1, 0, resample = 0, fill = 0),
    "resample"
  )

  old_fillcolor <- expect_warning(
    transform_affine(x, 0, c(0, 1), 1, 0, interpolation = 0, fillcolor = 0),
    "fillcolor"
  )

  new <- transform_affine(x, 0, c(0, 1), 1, 0, interpolation = 0, fill = 0)

  expect_equal_to_r(old_resample, as_array(new))
  expect_equal_to_r(old_fillcolor, as_array(new))

})

test_that("affine validates positive scale", {

  x <- torch_eye(8)$view(c(1, 1, 8, 8))
  expect_error(transform_affine(x, 0, c(0, 0), 0, 0), "positive")
  expect_error(transform_affine(x, 0, c(0, 0), 1, 0, interpolation = "bicubic"), "Unsupported interpolation mode")

})

test_that("linear transformation", {

  c <- 3
  h <- 24
  w <- 32

  tensor <- torch::torch_randn(c, h, w)
  matrix <- torch::torch_rand(c * h * w, c * h * w)
  mean_vector <- torch::torch_rand(c * h * w)

  out <- transform_linear_transformation(tensor, matrix, mean_vector)

  expect_equal(dim(out), c(3, 24, 32))

  outb <- transform_linear_transformation(tensor$unsqueeze(1), matrix, mean_vector)

  expect_equal(dim(outb), c(1, 3, 24, 32))
})

test_that("adjust hue", {

  hue_factor <- c(-0.45, -0.25, 0.0, 0.25, 0.45)
  x <- torch::torch_rand(3, 24, 32)

  for (f in hue_factor) {
    out <- transform_adjust_hue(x, f)
    expect_equal(dim(out), dim(x))
  }

  for (f in hue_factor) {
    out <- transform_adjust_hue(x$unsqueeze(1), f)
    expect_equal(dim(out), dim(x$unsqueeze(1)))
  }

})

test_that("grayscale", {

  x <- torch::torch_rand(3, 24, 32)
  out <- transform_grayscale(x, 3)
  expect_equal(dim(out), dim(x))
  expect_equal(dim(out)[1], 3)

  out <- transform_grayscale(x, 1)
  expect_equal(dim(out)[2:3], dim(x)[2:3])
  expect_equal(dim(out)[1], 1)

  outb <- transform_grayscale(x$unsqueeze(1), 3)
  expect_equal(dim(outb), dim(x$unsqueeze(1)))
  expect_equal(dim(outb)[2], 3)

  outb <- transform_grayscale(x$unsqueeze(1), 1)
  expect_equal(dim(outb)[3:4], dim(x)[2:3])
  expect_equal(dim(outb)[2], 1)

})

test_that("random grayscale", {

  tensor <- torch::torch_rand(3, 24, 32)
  for (p in seq(0, 1, length.out = 10)) {
    out <- transform_random_grayscale(tensor, p)
    expect_equal(dim(out), dim(tensor))
  }

  for (p in seq(0, 1, length.out = 10)) {
    outb <- transform_random_grayscale(tensor$unsqueeze(1), p)
    expect_equal(dim(outb), dim(tensor$unsqueeze(1)))
  }

})


test_that("random vertical flip", {

  tensor <- torch::torch_randn(3, 24, 32)

  for (i in 1:10) {
    out <- transform_random_vertical_flip(tensor)
    expect_equal(dim(out), dim(tensor))
  }
  for (p in seq(0, 1, length.out = 10)) {
    out <- transform_random_vertical_flip(tensor, p)
    expect_equal(dim(out), dim(tensor))
  }

  for (i in 1:10) {
    outb <- transform_random_vertical_flip(tensor$unsqueeze(1))
    expect_equal(dim(outb), dim(tensor$unsqueeze(1)))
  }
  for (p in seq(0, 1, length.out = 10)) {
    outb <- transform_random_vertical_flip(tensor$unsqueeze(1), p)
    expect_equal(dim(outb), dim(tensor$unsqueeze(1)))
  }
})


test_that("random rotation works", {

  x <- torch::torch_tensor(array(1, dim = c(3, 200, 200)))

  # Transforms
  rotate <- function(img) transform_random_rotation(img, 20)

  expect_error(rotate(x), regexp = NA)
  expect_error(rotate(x$unsqueeze(1)), regexp = NA)


})

test_that("random choice transform works", {

  # Example Image
  x <- torch_ones(c(3, 200, 200))

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
      x,
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

  expect_error(regexp = NA, {
    transform_random_choice(
      x$unsqueeze(1),
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


test_that("crop pads when the crop leaves the image", {
  x <- torch_arange(1, 24)$reshape(c(1, 4, 6))

  # fully inside: identical to indexing the image
  expect_equal_to_r(transform_crop(x, 2, 3, 2, 3), as_array(x[, 2:3, 3:5]))

  # partially outside: the requested size, the overlap unchanged, zeros elsewhere
  o <- transform_crop(x, top = 3, left = 5, height = 4, width = 4)
  expect_equal(dim(o), c(1, 4, 4))
  expect_equal_to_r(o[, 1:2, 1:2], as_array(x[, 3:4, 5:6]))
  expect_equal(as.numeric(o$sum()), as.numeric(x[, 3:4, 5:6]$sum()))

  # a start before the image pads on the top and the left
  o <- transform_crop(x, top = -1, left = 0, height = 4, width = 4)
  expect_equal(dim(o), c(1, 4, 4))
  expect_equal(as.numeric(o[, 1:2, ]$sum()), 0)          # the two padded rows
  expect_equal(as.numeric(o[, , 1]$sum()), 0)            # the padded column
  expect_equal_to_r(o[, 3:4, 2:4], as_array(x[, 1:2, 1:3]))

  # fully outside: all zeros, but still the requested size
  o <- transform_crop(x, top = 30, left = 30, height = 4, width = 4)
  expect_equal(dim(o), c(1, 4, 4))
  expect_equal(as.numeric(o$sum()), 0)

  # batch tensors behave the same way
  expect_equal(dim(transform_crop(x$unsqueeze(1), 3, 5, 4, 4)), c(1, 1, 4, 4))

  # the dtype is preserved, also on the padded path
  xi <- torch_ones(1, 2, 2, dtype = torch_long())
  expect_true(transform_crop(xi, 1, 1, 4, 4)$dtype == torch_long())
  expect_true(transform_crop(xi, 1, 1, 2, 2)$dtype == torch_long())
})

test_that("rgb_to_grayscale keeps the channel dimension", {
  x <- torch_rand(3, 4, 6)
  o <- transform_rgb_to_grayscale(x)
  expect_equal(dim(o), c(1, 4, 6))
  # the values are the usual luminance weights
  expect_equal(
    round(as_array(o[1, , ]), 5),
    round(as_array(0.2989 * x[1, , ] + 0.5870 * x[2, , ] + 0.1140 * x[3, , ]), 5)
  )
  expect_equal(dim(transform_rgb_to_grayscale(x$unsqueeze(1))), c(1, 1, 4, 6))

  # `transform_grayscale()` repeats that channel, also for batch tensors
  expect_equal(dim(transform_grayscale(x, num_output_channels = 1)), c(1, 4, 6))
  g3 <- transform_grayscale(x, num_output_channels = 3)
  expect_equal(dim(g3), c(3, 4, 6))
  # every repeated channel is the same grayscale image
  expect_equal_to_r(g3[2, , ], as_array(o[1, , ]))
  expect_equal_to_r(g3[3, , ], as_array(o[1, , ]))
  expect_equal(dim(transform_grayscale(x$unsqueeze(1), num_output_channels = 3)), c(1, 3, 4, 6))

  # the dtype is preserved
  xi <- torch_ones(3, 2, 2, dtype = torch_uint8())
  expect_true(transform_rgb_to_grayscale(xi)$dtype == torch_uint8())

  # the operators that build on it are unaffected by the extra dimension
  expect_equal(dim(transform_adjust_saturation(x, 1.5)), c(3, 4, 6))
  expect_equal(dim(transform_adjust_contrast(x, 1.5)), c(3, 4, 6))
})

test_that("rgb2hsv and hsv2rgb are inverse to each other", {
  x <- torch_rand(3, 8, 10)
  expect_equal(round(as_array(hsv2rgb(rgb2hsv(x))), 4), round(as_array(x), 4))

  b <- torch_rand(2, 3, 8, 10)
  expect_equal(round(as_array(hsv2rgb(rgb2hsv(b))), 4), round(as_array(b), 4))

  # known colours: (h, s, v) of pure red is (0, 1, 1), of pure blue (2/3, 1, 1)
  red <- torch_tensor(array(c(1, 0, 0), dim = c(3, 1, 1)))
  expect_equal(round(as.numeric(rgb2hsv(red)), 4), c(0, 1, 1))
  blue <- torch_tensor(array(c(0, 0, 1), dim = c(3, 1, 1)))
  expect_equal(round(as.numeric(rgb2hsv(blue)), 4), round(c(2 / 3, 1, 1), 4))
})

test_that("adjust_hue rotates the hue in both directions", {
  red <- torch_tensor(array(c(1, 0, 0), dim = c(3, 1, 1)))
  # a third of the hue wheel takes red to green, and back to blue in the other direction
  expect_equal(round(as.numeric(transform_adjust_hue(red, 1 / 3)), 3), c(0, 1, 0))
  expect_equal(round(as.numeric(transform_adjust_hue(red, -1 / 3)), 3), c(0, 0, 1))
  # half a turn is the same in both directions
  expect_equal(round(as.numeric(transform_adjust_hue(red, 0.5)), 3), c(0, 1, 1))
  expect_equal(round(as.numeric(transform_adjust_hue(red, -0.5)), 3), c(0, 1, 1))

  x <- torch_rand(3, 4, 6)
  # a hue factor of 0 leaves the image unchanged, and opposite rotations cancel out
  expect_equal(round(as_array(transform_adjust_hue(x, 0)), 4), round(as_array(x), 4))
  expect_equal(
    round(as_array(transform_adjust_hue(transform_adjust_hue(x, -0.5), 0.5)), 4),
    round(as_array(x), 4)
  )
  expect_equal(
    round(as_array(transform_adjust_hue(transform_adjust_hue(x, -0.25), -0.25)), 4),
    round(as_array(transform_adjust_hue(x, -0.5)), 4)
  )

  # uint8 images keep their dtype
  xi <- (x * 255)$to(dtype = torch_uint8())
  expect_true(transform_adjust_hue(xi, 0.2)$dtype == torch_uint8())
})

test_that("adjust_hue only accepts 1 or 3 channels", {
  x <- torch_rand(3, 4, 6)
  expect_equal(dim(transform_adjust_hue(x, 0.2)), c(3, 4, 6))

  # a single channel has no hue to adjust and is returned unchanged
  gray <- torch_rand(1, 4, 6)
  expect_equal_to_r(transform_adjust_hue(gray, 0.2), as_array(gray))
  expect_equal_to_r(transform_adjust_hue(gray$unsqueeze(1), 0.2), as_array(gray$unsqueeze(1)))

  expect_error(transform_adjust_hue(torch_rand(4, 4, 6), 0.2), "channel values are 1 or 3")
  expect_error(transform_adjust_hue(torch_rand(2, 2, 4, 6), 0.2), "channel values are 1 or 3")
  # `transform_color_jitter()` reaches the same check through its `hue` argument
  expect_error(transform_color_jitter(torch_rand(4, 4, 6), hue = 0.2), "channel values are 1 or 3")
})
