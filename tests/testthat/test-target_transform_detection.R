test_that("target_transform_resize with c(h, w) rescales boxes correctly", {
  # Image 100x200 (H x W), resize to 200x400 -> scale 2x
  target <- make_detection_target(boxes = matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  out <- target_transform_resize(target, c(200L, 400L))

  # scale_h = 200/100 = 2, scale_w = 400/200 = 2
  expected_boxes <- matrix(c(20, 40, 100, 120), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_tensor_dtype(out$boxes, torch_float32())
})

test_that("target_transform_resize with integer rescales proportionally", {
  # Image 100x200 (H x W), size = 50
  # max(100, 200) = 200, scale = 50/200 = 0.25
  # new_h = round(100 * 0.25) = 25, new_w = round(200 * 0.25) = 50
  target <- make_detection_target(boxes = matrix(c(40, 20, 80, 60), ncol = 4), image_size = c(100L, 200L))

  out <- target_transform_resize(target, 50L)

  # scale_h = 25/100 = 0.25, scale_w = 50/200 = 0.25
  expected_boxes <- matrix(c(10, 5, 20, 15), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 25L)
  expect_equal(out$image_width, 50L)
})

test_that("target_transform_resize updates original size", {
  target <- make_detection_target(boxes = matrix(c(0, 0, 10, 10), ncol = 4), image_size = c(100L, 100L))

  out <- target_transform_resize(target, c(300L, 400L))

  expect_equal(out$image_height, 300L)
  expect_equal(out$image_width, 400L)
})

test_that("target_transform_resize preserves other target fields", {
  target <- make_detection_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    labels = torch_tensor(c(3L, 5L), dtype = torch_long()),
    image_size = c(100L, 200L),
    image_id = 42L,
    area = torch_tensor(c(100, 200), dtype = torch_float32()),
    iscrowd = torch_tensor(c(0L, 1L), dtype = torch::torch_uint8())
  )

  out <- target_transform_resize(target, c(200L, 400L))

  # labels, image_id, area, iscrowd must remain unchanged
  expect_equal_to_r(out$labels, c(3L, 5L))
  expect_tensor_dtype(out$labels, torch_long())
  expect_equal_to_r(out$image_id, 42L)
  expect_equal_to_r(out$area, c(100, 200), tolerance = 1e-5)
  expect_tensor_dtype(out$area, torch_float32())
  expect_equal_to_r(out$iscrowd, as.raw(c(0, 1)))
  expect_tensor_dtype(out$iscrowd, torch::torch_uint8())
})

test_that("target_transform_resize handles non-uniform scaling", {
  # Image 100x200, resize to 300x200 -> scale_h = 3, scale_w = 1
  target <- make_detection_target(boxes = matrix(c(10, 20, 180, 70), ncol = 4), image_size = c(100L, 200L))

  out <- target_transform_resize(target, c(300L, 200L))

  # xmin=10*1=10, ymin=20*3=60, xmax=180*1=180, ymax=70*3=210
  expected_boxes <- matrix(c(10, 60, 180, 210), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize handles empty boxes", {
  target <- make_detection_target(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long()),
    image_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(200L, 400L))

  expect_equal(out$boxes$shape, c(0L, 4L))
  expect_tensor(out$boxes)
  expect_equal(out$image_height, 200L)
  expect_equal(out$image_width, 400L)
})

test_that("target_transform_resize handles multiple boxes", {
  target <- make_detection_target(
    boxes = matrix(
      c(10, 20, 50, 60, 0, 0, 100, 100, 25, 50, 75, 150),
      ncol = 4,
      byrow = TRUE
    ),
    image_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(200L, 400L))

  # scale_h = 2, scale_w = 2
  expected_boxes <- matrix(c(20, 40, 100, 120, 0, 0, 200, 200, 50, 100, 150, 300), ncol = 4, byrow = TRUE)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize handles boxes at image boundaries", {
  target <- make_detection_target(boxes = matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))

  out <- target_transform_resize(target, c(200L, 400L))

  expected_boxes <- matrix(c(0, 0, 400, 200), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize with integer handles square image", {
  # Square image 100x100, size = 50
  # max(100, 100) = 100, scale = 50/100 = 0.5
  # new_h = 50, new_w = 50
  target <- make_detection_target(boxes = matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 100L))

  out <- target_transform_resize(target, 50L)

  expected_boxes <- matrix(c(5, 10, 25, 30), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 50L)
  expect_equal(out$image_width, 50L)
})

test_that("target_transform_resize with integer handles portrait image", {
  # Portrait image 200x100 (H > W), size = 50
  # max(200, 100) = 200, scale = 50/200 = 0.25
  # new_h = 50, new_w = 25
  target <- make_detection_target(boxes = matrix(c(10, 40, 50, 120), ncol = 4), image_size = c(200L, 100L))

  out <- target_transform_resize(target, 50L)

  # scale_h = 50/200 = 0.25, scale_w = 25/100 = 0.25
  expected_boxes <- matrix(c(2.5, 10, 12.5, 30), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 50L)
  expect_equal(out$image_width, 25L)
})

test_that("target_transform_resize with integer handles landscape image", {
  # Landscape image 100x200 (W > H), size = 50
  # max(100, 200) = 200, scale = 50/200 = 0.25
  # new_h = 25, new_w = 50
  target <- make_detection_target(boxes = matrix(c(20, 10, 120, 50), ncol = 4), image_size = c(100L, 200L))

  out <- target_transform_resize(target, 50L)

  # scale_h = 25/100 = 0.25, scale_w = 50/200 = 0.25
  expected_boxes <- matrix(c(5, 2.5, 30, 12.5), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 25L)
  expect_equal(out$image_width, 50L)
})


test_that("target_transform_resize can be composed with pipe", {
  skip_if_not_installed("magrittr")

  target <- make_detection_target(boxes = matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  # Usage with the native pipe
  out <- target |>
    target_transform_resize(size = c(200L, 400L)) |>
    target_transform_resize(size = c(100L, 200L))

  expect_equal_to_r(out$boxes, as.matrix(target$boxes$cpu()), tolerance = 1e-4)
  expect_equal(out$image_height, target$image_height)
  expect_equal(out$image_width, target$image_width)

  # Usage with the magrittr pipe
  out <- target %>%
    target_transform_resize(size = c(200L, 400L)) %>%
    target_transform_resize(size = c(100L, 200L))

  expect_equal_to_r(out$boxes, as.matrix(target$boxes$cpu()), tolerance = 1e-4)
  expect_equal(out$image_height, target$image_height)
  expect_equal(out$image_width, target$image_width)
})

test_that("target_transform_resize errors when one of image size is missing", {
  target <- list(
    boxes = torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4)),
    labels = torch_ones(1L, dtype = torch_long())
  )

  expect_error(target_transform_resize(target, c(200L, 400L)), "image_width")
})

test_that("target_transform_resize does not mutate the input target", {
  target <- make_detection_target(boxes = matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  original_boxes <- as.matrix(target$boxes$cpu())
  original_height <- target$image_height
  original_width <- target$image_width

  out <- target_transform_resize(target, c(200L, 400L))

  # Original target must not be modified
  expect_equal_to_r(target$boxes, original_boxes)
  expect_equal(target$image_height, original_height)
  expect_equal(target$image_width, original_width)

  # Result must be different
  expect_false(identical(as.matrix(out$boxes$cpu()), original_boxes))
})

test_that("target_transform_rotate converts boxes to xyxyr format", {
  target <- make_detection_target(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- target_transform_rotate(target, angle = 0)

  expect_tensor_shape(result$boxes, c(1, 5))
  expect_tensor_dtype(result$boxes, torch_float())
  expect_equal_to_r(result$boxes[1, 1:4], c(10, 20, 50, 60))
  expect_equal_to_r(result$boxes[1, 5], 0)
})

test_that("target_transform_rotate applies non-zero rotation angle to boxes", {
  target <- make_detection_target(boxes = matrix(c(70, 40, 130, 160), ncol = 4), image_size = c(200L, 200L))
  result <- target_transform_rotate(target, angle = 73)

  cx <- 100
  cy <- 100
  hw <- 30
  hh <- 60
  expect_tensor_shape(result$boxes, c(1, 5))
  expect_tensor_dtype(result$boxes, torch_float())
  expect_equal_to_r(result$boxes[1, 1], cx - hw, tolerance = 1e-5)
  expect_equal_to_r(result$boxes[1, 3], cx + hw, tolerance = 1e-5)
  expect_equal_to_r(result$boxes[1, 5], 73, tolerance = 1e-5)

  angle_rad <- deg2rad(45)
  ct <- cos(angle_rad)
  st <- sin(angle_rad)
  corners_x <- c(cx - hw * ct + hh * st, cx + hw * ct + hh * st,
                 cx + hw * ct - hh * st, cx - hw * ct - hh * st)
  corners_y <- c(cy - hw * st - hh * ct, cy + hw * st - hh * ct,
                 cy + hw * st + hh * ct, cy - hw * st + hh * ct)
  expect_true(all(corners_x >= 0 & corners_x <= 200))
  expect_true(all(corners_y >= 0 & corners_y <= 200))
})

test_that("target_transform_rotate preserves labels and other target fields", {
  target <- make_detection_target(
    boxes = matrix(
      c(10, 20, 50, 60, 5, 5, 15, 25),
      ncol = 4,
      byrow = TRUE
    ),
    labels = torch_tensor(c(1L, 2L), dtype = torch_long())
  )
  original_labels <- target$labels$clone()

  result <- target_transform_rotate(target, angle = 0)

  expect_true(result$labels$eq(original_labels)$all()$item())
  expect_equal(result$image_height, 100L)
  expect_equal(result$image_width, 200L)
})

test_that("target_transform_rotate handles empty boxes", {
  target <- make_detection_target(boxes = matrix(numeric(0), ncol = 4), labels = torch_zeros(0L, dtype = torch_long()))
  result <- target_transform_rotate(target, angle = 27)

  expect_tensor_shape(result$boxes, c(0, 5))
  expect_tensor_dtype(result$boxes, torch_float())
})

test_that("target_transform_rotate handles multiple boxes", {
  boxes <- matrix(c(10, 20, 50, 60, 100, 200, 150, 250, 0, 0, 300, 400), ncol = 4, byrow = TRUE)
  target <- make_detection_target(boxes)
  result <- target_transform_rotate(target, angle = 442)

  expect_tensor_shape(result$boxes, c(3, 5))
  expect_tensor_dtype(result$boxes, torch_float())
  expect_equal_to_r(result$boxes[, 5], rep(442, 3L))
})

test_that("target_transform_rotate does not mutate input when (angle %% 180 == 0)", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  target <- make_detection_target(boxes)

  result <- target_transform_rotate(target, angle = 180)

  expect_equal_to_r(result$boxes, matrix(c(boxes, 180), ncol = 5))
  expect_tensor_shape(result$boxes, c(1, 5))

  result <- target_transform_rotate(target, angle = 360)

  expect_equal_to_r(result$boxes, matrix(c(boxes, 360), ncol = 5))
  expect_tensor_shape(result$boxes, c(1, 5))

  result <- target_transform_rotate(target, angle = -180)

  expect_equal_to_r(result$boxes, matrix(c(boxes, -180), ncol = 5))
  expect_tensor_shape(result$boxes, c(1, 5))
})

test_that("target_transform_rotate can be composed with pipe", {
  skip_if_not_installed("magrittr")

  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  target <- make_detection_target(boxes,
                                  image_size = c(100L, 200L))

  # Usage with the native pipe
  out <- target |>
    target_transform_rotate(angle = 15) |>
    target_transform_rotate(angle = 165)

  expect_equal_to_r(out$boxes, matrix(c(boxes, 180), ncol = 5))
  expect_equal(out$image_height, target$image_height)
  expect_equal(out$image_width, target$image_width)

  # Usage with the magrittr pipe
  out <- target %>%
    target_transform_rotate(angle = -165) %>%
    target_transform_rotate(angle = -15)

  expect_equal_to_r(out$boxes, out$boxes, matrix(c(boxes, -180), ncol = 5))
  expect_equal(out$image_height, target$image_height)
  expect_equal(out$image_width, target$image_width)
})


