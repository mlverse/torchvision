test_that("transform works for magick images", {
  im <- magick::image_read("assets/class/horse/horse-2.tif")

  im <- transform_random_resized_crop(im, size = c(224, 224))
  ii <- magick::image_info(im)
  expect_identical(c(ii$width, ii$height), c(224,224))
})
