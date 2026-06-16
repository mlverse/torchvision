test_that("transform works for magick images", {
  im <- magick::image_read("assets/class/horse/horse-2.tif")

  im <- transform_random_resized_crop(im, size = c(224, 224))
  ii <- magick::image_info(im)
  expect_equal(c(ii$width, ii$height), c(224,224))
})

test_that("transform_to_tensor works with magick image lists", {
  im <- magick::image_read("assets/class/horse/horse-2.tif")
  o <- transform_to_tensor(list(im, im, im))
  expect_tensor_shape(o, c(3, 3, 142, 180))
})

test_that("transform_sahi_crop works with magick images", {
  im <- magick::image_read("assets/class/horse/horse-2.tif")

  res <- transform_sahi_crop(im, size = c(50, 60), overlap_size_ratio = c(0, 0))

  expect_true("images" %in% names(res))
  expect_true("crop_windows" %in% names(res))
  expect_gt(length(res$images), 0)
  expect_s3_class(res$images[[1]], "magick-image")
  expect_type(res$crop_windows[[1]]$top, "double")
})
