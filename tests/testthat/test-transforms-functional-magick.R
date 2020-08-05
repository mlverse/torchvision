test_that("resize", {

  im <- magick::image_read("torch.png")
  r <- tfm_resize(im, 10)

  expect_equal(magick::image_info(r)$width, 10)

  r <- tfm_resize(im, c(10, 10))

  expect_equal(magick::image_info(r)$width, 10)
  expect_equal(magick::image_info(r)$height, 10)
})
