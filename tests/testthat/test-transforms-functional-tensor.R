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
