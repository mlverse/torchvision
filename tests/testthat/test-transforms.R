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
