# Helper to build a test target
make_target <- function(boxes, labels = NULL, orig_size = c(100L, 200L), image_id = 1L, area = NULL, iscrowd = NULL) {
  if (is.matrix(boxes)) {
    boxes <- torch_tensor(boxes, dtype = torch_float32())
  }
  if (is.null(labels)) {
    labels <- torch_ones(boxes$size(1), dtype = torch_long())
  }
  if (is.null(area)) {
    area <- (boxes[, 3] - boxes[, 1]) * (boxes[, 4] - boxes[, 2])
  }
  if (is.null(iscrowd)) {
    iscrowd <- torch_zeros(boxes$size(1), dtype = torch_uint8())
  }
  list(
    boxes = boxes,
    labels = labels,
    image_height = orig_size[1],
    image_width = orig_size[2],
    image_id = torch_tensor(image_id, dtype = torch_long()),
    area = area,
    iscrowd = iscrowd
  )
}

test_that("target_transform_resize with c(h, w) rescales boxes correctly", {
  # Image 100x200 (H x W), resize to 200x400 -> scale 2x
  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    orig_size = c(100L, 200L)
  )

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
  target <- make_target(
    boxes = matrix(c(40, 20, 80, 60), ncol = 4),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, 50L)

  # scale_h = 25/100 = 0.25, scale_w = 50/200 = 0.25
  expected_boxes <- matrix(c(10, 5, 20, 15), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 25L)
  expect_equal(out$image_width, 50L)
})

test_that("target_transform_resize updates original size", {
  target <- make_target(
    boxes = matrix(c(0, 0, 10, 10), ncol = 4),
    orig_size = c(100L, 100L)
  )

  out <- target_transform_resize(target, c(300L, 400L))

  expect_equal(out$image_height, 300L)
  expect_equal(out$image_width, 400L)
})

test_that("target_transform_resize preserves other target fields", {
  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    labels = torch_tensor(c(3L, 5L), dtype = torch_long()),
    orig_size = c(100L, 200L),
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
  target <- make_target(
    boxes = matrix(c(10, 20, 180, 70), ncol = 4),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(300L, 200L))

  # xmin=10*1=10, ymin=20*3=60, xmax=180*1=180, ymax=70*3=210
  expected_boxes <- matrix(c(10, 60, 180, 210), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize handles empty boxes", {
  target <- make_target(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long()),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(200L, 400L))

  expect_equal(out$boxes$shape, c(0L, 4L))
  expect_tensor(out$boxes)
  expect_equal(out$image_height, 200L)
  expect_equal(out$image_width, 400L)
})

test_that("target_transform_resize handles multiple boxes", {
  target <- make_target(
    boxes = matrix(
      c(
        10,
        20,
        50,
        60,
        0,
        0,
        100,
        100,
        25,
        50,
        75,
        150
      ),
      ncol = 4,
      byrow = TRUE
    ),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(200L, 400L))

  # scale_h = 2, scale_w = 2
  expected_boxes <- matrix(
    c(
      20,
      40,
      100,
      120,
      0,
      0,
      200,
      200,
      50,
      100,
      150,
      300
    ),
    ncol = 4,
    byrow = TRUE
  )
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize handles boxes at image boundaries", {
  target <- make_target(
    boxes = matrix(c(0, 0, 200, 100), ncol = 4),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, c(200L, 400L))

  expected_boxes <- matrix(c(0, 0, 400, 200), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("target_transform_resize with integer handles square image", {
  # Square image 100x100, size = 50
  # max(100, 100) = 100, scale = 50/100 = 0.5
  # new_h = 50, new_w = 50
  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    orig_size = c(100L, 100L)
  )

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
  target <- make_target(
    boxes = matrix(c(10, 40, 50, 120), ncol = 4),
    orig_size = c(200L, 100L)
  )

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
  target <- make_target(
    boxes = matrix(c(20, 10, 120, 50), ncol = 4),
    orig_size = c(100L, 200L)
  )

  out <- target_transform_resize(target, 50L)

  # scale_h = 25/100 = 0.25, scale_w = 50/200 = 0.25
  expected_boxes <- matrix(c(5, 2.5, 30, 12.5), ncol = 4)
  expect_equal_to_r(out$boxes, expected_boxes, tolerance = 1e-5)
  expect_equal(out$image_height, 25L)
  expect_equal(out$image_width, 50L)
})

test_that("target_transform_resize is composable in a pipeline", {
  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    orig_size = c(100L, 200L)
  )

  # Pipeline: resize 2x then resize 0.5x -> back to original
  out1 <- target_transform_resize(target, c(200L, 400L))
  out2 <- target_transform_resize(out1, c(100L, 200L))

  expect_equal_to_r(out2$boxes, as.matrix(target$boxes$cpu()), tolerance = 1e-4)
  expect_equal(out2$image_height, target$image_height)
  expect_equal(out2$image_width, target$image_width)
})

test_that("target_transform_resize works with pipe", {
  skip_if_not_installed("magrittr")

  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    orig_size = c(100L, 200L)
  )

  # Usage with the magrittr pipe
  out <- target |>
    target_transform_resize(size = c(200L, 400L)) |>
    target_transform_resize(size = c(100L, 200L))

  expect_equal_to_r(out$boxes, as.matrix(target$boxes$cpu()), tolerance = 1e-4)
  expect_equal(out$image_height, target$image_height)
  expect_equal(out$image_width, target$image_width)

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

  expect_error(
    target_transform_resize(target, c(200L, 400L)),
    "image_width"
  )
})

test_that("target_transform_resize does not mutate the input target", {
  target <- make_target(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    orig_size = c(100L, 200L)
  )

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

# item_transform_rotate ---------------------------------------------------

make_item <- function(boxes, labels = NULL, image_size = c(100L, 200L)) {
  if (is.matrix(boxes)) {
    boxes <- torch_tensor(boxes, dtype = torch_float32())
  }
  if (is.null(labels)) {
    labels <- torch_ones(boxes$size(1), dtype = torch_long())
  }
  x <- torch_randn(3, image_size[1], image_size[2])
  y <- list(
    boxes = boxes,
    labels = labels,
    image_height = image_size[1],
    image_width = image_size[2]
  )
  item <- list(x = x, y = y)
  class(item) <- c("image_with_bounding_box", "list")
  item
}

test_that("item_transform_rotate converts image_with_bounding_box to image_with_rotated_box", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 0)

  expect_true(inherits(result, "image_with_rotated_box"))
  expect_equal(result$y$boxes$size(2), 5L)
  expect_equal(torch::as_array(result$y$boxes[1, 1:4]), c(10, 20, 50, 60))
  expect_equal(as.numeric(result$y$boxes[1, 5]$cpu()), 0)
})

test_that("item_transform_rotate rotates both image and bounding boxes", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_x <- item$x$clone()
  result <- item_transform_rotate(item, angle = 90)

  # Image should be rotated (not identical to original)
  expect_false(result$x$eq(original_x)$all()$item())
  # With expand=FALSE, the canvas size stays the same
  expect_equal(result$x$size(-2), 100L)
  expect_equal(result$x$size(-1), 200L)
  # Boxes should be in xyxyr format
  expect_equal(result$y$boxes$size(2), 5L)
  expect_equal(as.numeric(result$y$boxes[1, 5]$cpu()), 90)
})

test_that("item_transform_rotate preserves labels and other target fields", {
  item <- make_item(
    boxes = matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = torch_tensor(c(1L, 2L), dtype = torch_long())
  )
  original_labels <- item$y$labels$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_rotate handles empty boxes", {
  item <- make_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_rotate(item, angle = 0)

  expect_true(inherits(result, "image_with_rotated_box"))
  expect_equal(result$y$boxes$size(1), 0L)
  expect_equal(result$y$boxes$size(2), 5L)
})

test_that("item_transform_rotate handles multiple boxes with angle=0", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  item <- make_item(boxes)
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$y$boxes$size(1), 3L)
  expect_equal(result$y$boxes$size(2), 5L)
  expect_equal(torch::as_array(result$y$boxes[, 5]), c(0, 0, 0))
})

test_that("item_transform_rotate does not mutate input", {
  boxes <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  item <- make_item(boxes)
  original_boxes <- boxes$clone()
  original_x <- item$x$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_equal_to_r(item$y$boxes, as.matrix(original_boxes$cpu()))
  expect_equal(item$y$boxes$size(2), 4L)
  expect_false(inherits(item, "image_with_rotated_box"))
  expect_true(item$x$eq(original_x)$all()$item())
})

test_that("item_transform_rotate applies non-zero rotation angle to boxes", {
  # A 2x2 square centered at (101, 101) in image coords: (100,100,102,102)
  # Rotating by 45 degrees should expand the enclosing box
  item <- make_item(boxes = matrix(c(100, 100, 102, 102), ncol = 4))
  result <- item_transform_rotate(item, angle = 45)

  # The enclosing box expands because corners rotate
  cx <- 101; cy <- 101
  half_diag <- sqrt(2)
  expect_equal(as.numeric(result$y$boxes[1, 1]$cpu()), cx - half_diag, tolerance = 1e-5)
  expect_equal(as.numeric(result$y$boxes[1, 3]$cpu()), cx + half_diag, tolerance = 1e-5)
  expect_equal(as.numeric(result$y$boxes[1, 5]$cpu()), 45, tolerance = 1e-5)
})

test_that("item_transform_rotate with expand=TRUE enlarges canvas", {
  item <- make_item(
    boxes = matrix(c(10, 20, 50, 60), ncol = 4),
    image_size = c(100L, 100L)
  )
  result <- item_transform_rotate(item, angle = 45, expand = TRUE)

  # Canvas should be larger than original
  new_h <- result$x$size(-2)
  new_w <- result$x$size(-1)
  expect_true(new_h > 100L)
  expect_true(new_w > 100L)

  # Boxes should be translated to new canvas coordinates
  expect_true(all(as.numeric(result$y$boxes[, 1]$cpu()) >= 0))
  expect_true(all(as.numeric(result$y$boxes[, 3]$cpu()) <= new_w))
})
