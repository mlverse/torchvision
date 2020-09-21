test_that("multiplication works", {
  im <- magick::image_read("tests/testthat/torch.png")
  transform_crop(im, 1, 1, 500, 500)

  im <- transform_random_resized_crop(im, size = c(224, 224))
})
