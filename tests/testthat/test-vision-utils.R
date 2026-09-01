context("vision-utils")

test_that("vision_make_grid works with 4D batch tensor", {
  images <- torch::torch_randn(c(4, 3, 16, 16))
  grid <- vision_make_grid(images, num_rows = 2, padding = 0)
  expect_tensor_shape(grid, c(3, 32, 32))
  expect_equal_to_r(grid$max() - grid$min(), 1, tolerance = 1e-4)
})

test_that("vision_make_grid works with multiple 3D tensors in ...", {
  imgs <- lapply(1:4, function(i) torch::torch_randn(c(3, 16, 16)))
  grid <- vision_make_grid(imgs[[1]], imgs[[2]], imgs[[3]], imgs[[4]], num_rows = 2, padding = 0)
  expect_tensor_shape(grid, c(3, 32, 32))
  expect_equal_to_r(grid$max() - grid$min(), 1, tolerance = 1e-4)
})

test_that("vision_make_grid works with multiple 4D tensors in ...", {
  batch1 <- torch::torch_randn(c(2, 3, 16, 16))
  batch2 <- torch::torch_randn(c(2, 3, 16, 16))
  grid <- vision_make_grid(batch1, batch2, num_rows = 2, padding = 0)
  expect_tensor_shape(grid, c(3, 32, 32))
})

test_that("vision_make_grid normalizes mixed uint8/float inputs to float [0,1]", {
  img_float <- torch::torch_rand(c(3, 16, 16))
  img_uint8 <- torch::torch_randint(0L, 256L, size = c(3, 16, 16))$to(torch::torch_uint8())
  grid <- vision_make_grid(img_float, img_uint8, num_rows = 1, padding = 0, scale = FALSE)
  expect_tensor_shape(grid, c(3, 16, 32))
  expect_tensor_dtype(grid, torch::torch_float32())
  expect_true(grid$min()$item() >= 0 && grid$max()$item() <= 1)
})

test_that("vision_make_grid errors on mixed 3D/4D tensors in ...", {
  t3d <- torch::torch_randn(c(3, 16, 16))
  t4d <- torch::torch_randn(c(2, 3, 16, 16))
  expect_error(vision_make_grid(t3d, t4d), class = "value_error")
})

test_that("vision_make_grid works with magick-image", {
  skip_if_not_installed("magick")
  imgs <- magick::image_read(rep(system.file("img", "Rlogo.png", package = "png"), 4))
  h <- magick::image_info(imgs[1])$height
  w <- magick::image_info(imgs[1])$width
  grid <- vision_make_grid(imgs, num_rows = 2, padding = 0)
  expect_tensor_shape(grid, c(3L, 2L * h, 2L * w))
})

test_that("vision_make_grid errors on unsupported type", {
  expect_error(vision_make_grid(list(1, 2, 3)), class = "cli_error")
})

test_that("draw_bounding_boxes works", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- (255 - (torch::torch_randint(low = 1, high = 60, size = c(3, 360, 360))))$to(torch::torch_uint8())
  x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
  y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
  w <- torch::torch_randint(low = 10, high = 100, size = c(12,1))
  h <- torch::torch_randint(low = 30, high = 60, size = c(12,1))
  boxes <- torch::torch_cat(c(x, y, x + w, y +  h), dim = 2)

  expect_error(bboxed_image <- draw_bounding_boxes(image_uint$to(dtype = torch::torch_int32()), boxes),
               class = "type_error", regexp = "torch_uint8")

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_float, boxes, labels = "dog", width = 5))
  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, labels = "Leptailurus serval constantina", width = 1))
  expect_tensor_dtype(bboxed_image, torch::torch_uint8())
  expect_tensor_shape(bboxed_image, c(3, 360, 360))

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, colors = "black", fill = TRUE))
})

test_that("draw_bounding_boxes correctly mask a complete image", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 1, high = 240, size = c(3, 360, 360))$to(torch::torch_uint8())
  boxes <- torch::torch_tensor(c(0,0,360,360))$unsqueeze(1)

  expect_no_error(bboxed_image <- draw_bounding_boxes(image_float, boxes, colors = "black", fill = TRUE))
  expect_no_error(bboxed_image <- draw_bounding_boxes(image_uint, boxes, colors = "black", fill = TRUE))
  # some invisible glitch remains
  expect_lte(bboxed_image$sum() %>% as.numeric, 3000)


})

test_that("draw_bounding_boxes draws each rotated box as a separate tight rectangle", {

  H <- 200
  W <- 300
  image <- torch::torch_zeros(3, H, W)
  boxes <- torch::torch_tensor(rbind(
    c(60, 60, 90, 90),
    c(210, 110, 240, 140)
  ), dtype = torch::torch_float32())

  item <- list(x = image, y = list(boxes = boxes, labels = c("a", "b"),
                                   image_height = H, image_width = W))
  class(item) <- c("image_with_bounding_box", "list")

  rotated <- item_transform_rotate(item, angle = 30, expand = FALSE)
  drawn <- draw_bounding_boxes(rotated, colors = "red", width = 3)
  red <- as_array(drawn[1, , ]$to(dtype = torch::torch_float()))

  boxes_r <- as.matrix(rotated$y$boxes$to(device = "cpu"))
  xmin <- boxes_r[, 1]; ymin <- boxes_r[, 2]
  xmax <- boxes_r[, 3]; ymax <- boxes_r[, 4]
  theta <- boxes_r[, 5]
  cx <- (xmin + xmax) / 2; cy <- (ymin + ymax) / 2
  hw <- (xmax - xmin) / 2; hh <- (ymax - ymin) / 2
  theta_rad <- deg2rad(theta)
  ct <- cos(theta_rad); st <- -sin(theta_rad)
  all_x <- cbind(cx - hw * ct + hh * st, cx + hw * ct + hh * st,
                 cx + hw * ct - hh * st, cx - hw * ct - hh * st)
  all_y <- cbind(cy - hw * st - hh * ct, cy + hw * st - hh * ct,
                 cy + hw * st + hh * ct, cy - hw * st + hh * ct)

  read_px <- function(pts) {
    x <- round(pts[1]); y <- round(pts[2])
    red[y + 1, x + 1]
  }

  # the border of each box must pass through the midpoint of each of its edges
  for (j in 1:2) {
    corners <- cbind(all_x[j, ], all_y[j, ])
    for (e in 1:4) {
      a <- corners[e, ]
      nxt <- corners[ifelse(e == 4, 1, e + 1), ]
      expect_gt(read_px((a + nxt) / 2), 200)
    }
  }

  # no border may be drawn along the straight lines joining one box to the other
  for (e in 1:4) {
    a <- c(all_x[1, e], all_y[1, e])
    b <- c(all_x[2, e], all_y[2, e])
    expect_lt(read_px((a + b) / 2), 100)
  }
})

test_that("draw_segmentation_masks works with boolean mask", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 190, high = 255, size = c(3, 360, 360))$to(torch::torch_uint8())
  lower_mask <- torch::torch_tril(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch_bool())
  upper_mask <- torch::torch_triu(torch::torch_ones(c(360, 360)), diagonal = FALSE)$to(torch_bool())
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

test_that("draw_segmentation_masks works with float mask", {

  image_float <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
  image_uint <- torch::torch_randint(low = 190, high = 255, size = c(3, 360, 360))$to(torch::torch_uint8())
  lower_mask <- torch::torch_tril(torch::torch_ones(c(360, 360)), diagonal = FALSE)
  upper_mask <- torch::torch_triu(torch::torch_ones(c(360, 360))*2, diagonal = FALSE)
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
  expect_error(tensor_image_browse(image), "Pass individual `image`, not batches")
  image <- torch::torch_randint(low = 1, high = 200, size = c(4, 360, 360))$to(torch::torch_uint8())
  expect_error(tensor_image_browse(image), "Only grayscale and RGB")

})

# ==== COCO sample drawing ====

test_that("draw_bounding_boxes works with coco_detection_sample", {
  skip_if_not(torch::torch_is_installed())

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
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

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
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

