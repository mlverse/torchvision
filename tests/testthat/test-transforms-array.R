test_that("transform_to_tensor works for 2D, 3D, 4D array", {
  x <- array(1L:120L, dim = c(10,12))
  o <- transform_to_tensor(x)
  expect_tensor_shape(o, c(1,10,12))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)

  x <- array(1L:120L, dim = c(1, 10,12))
  o <- transform_to_tensor(x)
  expect_tensor_shape(o, c(1,10,12))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)

  x <- array(1L:120L, dim = c(3, 5, 8))
  o <- transform_to_tensor(x)
  expect_tensor_shape(o, c(3, 5, 8))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)


  x <- array(1L:120L, dim = c(2, 1, 4,5))
  ob <- transform_to_tensor(x)
  expect_tensor_shape(ob, c(2, 1, 4, 5))
  expect_tensor_dtype(ob, torch_float())

  x <- array(1L:120L, dim = c(2, 3, 10,12))
  ob <- transform_to_tensor(x)
  expect_tensor_shape(ob, c(2, 3,10,12))
  expect_tensor_dtype(ob, torch_float())

})

test_that("transform_to_tensor works for list of arrays", {
  x <- array(1L:120L, dim = c(10,12))
  o <- transform_to_tensor(list(x, x))
  expect_tensor_shape(o, c(2,1,10,12))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)

  x <- array(1L:120L, dim = c(1, 10,12))
  o <- transform_to_tensor(list(x, x))
  expect_tensor_shape(o, c(2,1,10,12))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)

  x <- array(1L:120L, dim = c(3, 5, 8))
  o <- transform_to_tensor(list(x, x, x))
  expect_tensor_shape(o, c(3, 3, 5, 8))
  expect_tensor_dtype(o, torch_float())
  expect_equal_to_r(o$max(), 120/255)


  x <- array(1L:120L, dim = c(2, 1, 4,5))
  expect_error(transform_to_tensor(list(x,x)),
               "3D arrays")

})

test_that("transform_sahi_crop converts array inputs via default dispatch", {

  arr <- array(sample(0:255, 3 * 80 * 60, replace = TRUE), dim = c(80, 60, 3))

  res <- transform_sahi_crop(arr, size = c(40, 30), overlap_size_ratio = c(0.5, 0.5))

  expect_true("images" %in% names(res))
  expect_true("crop_windows" %in% names(res))
  expect_gt(length(res$images), 0)
  expect_tensor(res$images[[1]])
  expect_equal(res$images[[1]]$shape[[1]], 3)

})
