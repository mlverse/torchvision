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

  sp <- prepare_sahi_split(im, size = c(50, 60), overlap_size_ratio = c(0, 0))
  res <- transform_sahi_crop(im, sp)

  expect_s3_class(res, "magick-image")
  expect_length(res, 9)
})

test_that("transform_sahi_crop works with batched multi-frame magick images", {
  im <- magick::image_read("assets/class/horse/horse-2.tif")
  im_batch <- magick::image_join(c(im, im))

  sp <- prepare_sahi_split(im, size = c(50, 60), overlap_size_ratio = c(0, 0))
  res <- transform_sahi_crop(im_batch, sp)

  expect_s3_class(res, "magick-image")
  expect_length(res, 9 * 2)
})
