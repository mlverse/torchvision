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

test_that("draw_bounding_boxes correctly mask a complete image", {

  image <- torch::torch_randint(low = 1, high = 240, size = c(3, 360, 360))$to(torch::torch_uint8())
  boxes <- torch::torch_tensor(c(0,0,360,360))$unsqueeze(1)

  expect_no_error(bboxed_image <- draw_bounding_boxes(image, boxes, color = "black", fill = TRUE))
  # some invisible glitch remaains
  expect_lte(bboxed_image$sum() %>% as.numeric, 3000)
})

test_that("draw_segmentation_masks works", {

  image <- (255 - (torch::torch_randint(low = 1, high = 60, size = c(3, 360, 360))))$to(torch::torch_uint8())
  lower_mask <- torch::torch_tril(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch::torch_bool())
  upper_mask <- torch::torch_triu(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch::torch_bool())
  masks <- torch::torch_stack(c(lower_mask, upper_mask), dim = 1)

  expect_no_error(masked_image <- draw_segmentation_masks(image, masks))
  expect_tensor_dtype(masked_image, torch::torch_uint8())
  expect_tensor_shape(masked_image, c(3, 360, 360))

  color <-  data.frame("R" = runif(2,1,255), "G" = runif(2,1,255), "B" = runif(2,1,255))
  expect_no_error(masked_image <- draw_segmentation_masks(image, masks, color = color, alpha = 0.5))
})

test_that("plot works", {

  image <- (255 - (torch::torch_randint(low = 1, high = 200, size = c(3, 360, 360))))$to(torch::torch_uint8())
  expect_no_error(plot(image))
})

