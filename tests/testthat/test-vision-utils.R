context("vision-utils")

test_that("vision_make_grid", {

  images <- torch::torch_randn(c(4, 3, 16, 16))

  grid <- vision_make_grid(images, num_rows = 2, padding = 0)


  expect_equal(grid$size(), c(3, 32, 32))
  expect_equal(as.numeric(grid$max() - grid$min()), 1, tolerance = 1e-4)

})

test_that("draw_bounding_boxes works", {

  image1 <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image2 <- (255 - (torch::torch_randint(low = 1, high = 60, size = c(3, 360, 360))))$to(torch::torch_uint8())
  x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
  y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
  boxes <- torch::torch_cat(c(x, y, x + runif(1, 5, 60), y +  runif(1, 5, 10)), dim = 2)

  expect_error(bboxed_image <- draw_bounding_boxes(image1, boxes), "uint8")

  expect_no_error(bboxed_image <- draw_bounding_boxes(image2, boxes))
  expect_tensor_dtype(bboxed_image, torch::torch_uint8())
  expect_tensor_shape(bboxed_image, c(3, 360, 360))

  expect_no_error(bboxed_image <- draw_bounding_boxes(image2, boxes, color = "black", fill = TRUE))
})
