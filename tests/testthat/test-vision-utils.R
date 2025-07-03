context("vision-utils")

test_that("vision_make_grid", {

  images <- torch::torch_randn(c(4, 3, 16, 16))

  grid <- vision_make_grid(images, num_rows = 2, padding = 0)


  expect_equal(grid$size(), c(3, 32, 32))
  expect_equal(as.numeric(grid$max() - grid$min()), 1, tolerance = 1e-4)

})

test_that("draw_bounding_boxes works", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- (255 - (torch::torch_randint(low = 1, high = 60, size = c(3, 360, 360))))$to(torch::torch_uint8())
  x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
  y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
  boxes <- torch::torch_cat(c(x, y, x + runif(1, 5, 60), y +  runif(1, 5, 10)), dim = 2)

  expect_error(bboxed_image <- draw_bounding_boxes(image_uint$to(dtype = torch::torch_int32()), boxes),
               class = "type_error", regexp = "torch_uint8")

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_float, boxes, labels = "dog"))
  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, labels = "dog"))
  expect_tensor_dtype(bboxed_image, torch::torch_uint8())
  expect_tensor_shape(bboxed_image, c(3, 360, 360))

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, color = "black", fill = TRUE))
})

test_that("draw_bounding_boxes correctly mask a complete image", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 1, high = 240, size = c(3, 360, 360))$to(torch::torch_uint8())
  boxes <- torch::torch_tensor(c(0,0,360,360))$unsqueeze(1)

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_float, boxes, color = "black", fill = TRUE))
  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, color = "black", fill = TRUE))
  # some invisible glitch remains
  expect_lte(bboxed_image$sum() %>% as.numeric, 3000)


})

test_that("draw_segmentation_masks works", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 190, high = 255, size = c(3, 360, 360))$to(torch::torch_uint8())
  lower_mask <- torch::torch_tril(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch::torch_bool())
  upper_mask <- torch::torch_triu(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch::torch_bool())
  masks <- torch::torch_stack(c(lower_mask, upper_mask), dim = 1)

  expect_no_error(masked_image <- draw_segmentation_masks(image_float, masks))
  expect_tensor_dtype(masked_image, torch::torch_uint8())
  expect_tensor_shape(masked_image, c(3, 360, 360))

  expect_no_error(masked_image <- draw_segmentation_masks(image_uint, masks))
  expect_tensor_dtype(masked_image, torch::torch_uint8())
  expect_tensor_shape(masked_image, c(3, 360, 360))

  colors <-  c("navyblue", "orange3")
  expect_no_error(masked_image <- draw_segmentation_masks(image_uint, masks, colors = colors, alpha = 0.5))
})

test_that("draw_keypoints works", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 190, high = 255, size = c(3, 360, 360))$to(torch::torch_uint8())
  keypoints <- torch::torch_randint(low = 60, high = 300, size = c(4, 5, 2))
  colors <-  hcl.colors(n = 5)

  expect_no_error(keypoint_image <- draw_keypoints(image_float, keypoints))
  expect_tensor_dtype(keypoint_image, torch::torch_uint8())
  expect_tensor_shape(keypoint_image, c(3, 360, 360))
  expect_no_error(keypoint_image <- draw_keypoints(image_float, keypoints, colors = colors, radius = 7))

  expect_no_error(keypoint_image <- draw_keypoints(image_uint, keypoints))
  expect_tensor_dtype(keypoint_image, torch::torch_uint8())
  expect_tensor_shape(keypoint_image, c(3, 360, 360))
  expect_no_error(keypoint_image <- draw_keypoints(image_uint, keypoints, colors = colors, radius = 7))
})

test_that("tensor_image_browse works", {
  skip_on_cran()
  skip_on_ci()
  # uint8 color image
  image <- (255 - (torch::torch_randint(low = 1, high = 200, size = c(3, 360, 360))))$to(torch::torch_uint8())
  expect_no_error(tensor_image_browse(image))
  # uint8 grayscale image
  image <- (255 - (torch::torch_randint(low = 1, high = 200, size = c(1, 360, 360))))$to(torch::torch_uint8())
  expect_no_error(tensor_image_browse(image))

  # float color image
  image <- torch::torch_rand(size = c(3, 360, 360))
  expect_no_error(tensor_image_browse(image))
  # float grayscale image
  image <- torch::torch_rand(size = c(1, 360, 360))
  expect_no_error(tensor_image_browse(image))

  # error cases : shape
  image <- torch::torch_randint(low = 1, high = 200, size = c(4, 3, 360, 360))$to(torch::torch_uint8())
  expect_error(tensor_image_browse(image), "Pass individual `image`")
  image <- torch::torch_randint(low = 1, high = 200, size = c(4, 360, 360))$to(torch::torch_uint8())
  expect_error(tensor_image_browse(image), "Only grayscale and RGB")

})

# ==== COCO sample drawing ====

test_that("draw_bounding_boxes works with coco_detection_sample", {
  skip_if_not(torch::torch_is_installed())

  skip_if(Sys.getenv("TEST_LARGE_DATASETS") != "1",
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- coco_detection_dataset(root = "~/data", train = FALSE, year = "2017", download = TRUE)
  item <- ds[3]

  out <- draw_bounding_boxes(item)
  expect_tensor(out)
  expect_equal(out$ndim, 3)
  expect_equal(out$shape[1], 3)  # 3 color channels
  expect_gt(out$shape[2], 100)   # image height is reasonable
  expect_gt(out$shape[3], 100)   # image width is reasonable
})

test_that("draw_segmentation_masks works with coco_detection_sample", {
  skip_if_not(torch::torch_is_installed())

  skip_if(Sys.getenv("TEST_LARGE_DATASETS") != "1",
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- coco_detection_dataset(root = "~/data", train = FALSE, year = "2017", download = TRUE)
  item <- ds[3]

  if (item$y$masks$size(1) > 0) {
    out <- draw_segmentation_masks(item)
    expect_tensor(out)
    expect_equal(out$ndim, 3)
    expect_equal(out$shape[1], 3)
    expect_gt(out$shape[2], 100)
    expect_gt(out$shape[3], 100)
  } else {
    skip("No masks in this item — skipping mask drawing test.")
  }
})

