test_that("item_transform_rotate rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_rotate(img, angle = 45),
    "requires a dataset item"
  )
})

test_that("item_transform_rotate rejects numeric input", {
  expect_error(
    item_transform_rotate(42, angle = 0),
    "requires a dataset item"
  )
})

test_that("item_transform_rotate 0 degrees preserves image and boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100, 200))
  result <- item_transform_rotate(item, angle = 0)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal_to_r(result$x, as_array(item$x), tolerance = 1e-5)
  expect_equal(result$y$boxes$shape, c(1, 5))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes)
  expect_equal_to_r(result$y$boxes[, 5], 0)
})

test_that("item_transform_rotate expands canvas for non-axis-aligned angles", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(300L, 500L))
  result <- item_transform_rotate(item, angle = 30)

  # Read expected dimensions from actual output
  expect_equal(result$x$shape[1], 3)
  expect_equal(result$x$shape[2], result$y$image_height)  # H
  expect_equal(result$x$shape[3], result$y$image_width)   # W
})

test_that("item_transform_rotate 90 degrees swaps dimensions", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_rotate(item, angle = 90)

  # After 90° rotation with expand=TRUE, H and W swap
  expect_equal(result$x$shape, c(3, 400L, 200L))  # (C, new_H, new_W)
  expect_equal(result$y$image_height, 400L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_rotate 180 degrees preserves dimensions", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_rotate(item, angle = 180)

  expect_equal(result$x$shape, c(3, 200, 400))
  expect_equal(result$y$image_height, 200L)
  expect_equal(result$y$image_width, 400L)
})

test_that("item_transform_rotate negative angles work", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_rotate(item, angle = -45)

  expect_equal(result$x$shape[1], 3)
  expect_equal(result$x$shape[2],  result$y$image_height)
  expect_equal(result$x$shape[3],  result$y$image_width)
})

test_that("item_transform_rotate boxes are shifted for expanded canvas", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(300L, 500L))
  result <- item_transform_rotate(item, angle = 30)

  # 1. Calculer la géométrie attendue manuellement
  orig_box <- c(10, 20, 50, 60)
  angle <- 30

  cx <- (orig_box[1] + orig_box[3]) / 2  # 30
  cy <- (orig_box[2] + orig_box[4]) / 2  # 40
  hw <- (orig_box[3] - orig_box[1]) / 2  # 20
  hh <- (orig_box[4] - orig_box[2]) / 2  # 20

  # Nouvelle demi-dimension après rotation de 30°
  angle_rad <- angle * pi / 180
  new_hw <- hw * abs(cos(angle_rad)) + hh * abs(sin(angle_rad)) # ~27.32
  new_hh <- hw * abs(sin(angle_rad)) + hh * abs(cos(angle_rad)) # ~27.32

  # Expansion du canvas
  new_H <- as.integer(ceiling(500 * abs(sin(angle_rad)) + 300 * abs(cos(angle_rad))))
  new_W <- as.integer(ceiling(500 * abs(cos(angle_rad)) + 300 * abs(sin(angle_rad))))
  dx <- (new_W - 500) / 2
  dy <- (new_H - 300) / 2

  # Coordonnées attendues : centre shifté +/- nouvelle demi-dimension
  expected_xmin <- cx + dx - new_hw
  expected_xmax <- cx + dx + new_hw
  expected_ymin <- cy + dy - new_hh
  expected_ymax <- cy + dy + new_hh

  # 2. Vérification avec une tolérance réaliste pour l'arithmétique flottante
  expect_equal_to_r(result$y$boxes[1, 1], expected_xmin, tolerance = 1e-4)
  expect_equal_to_r(result$y$boxes[1, 3], expected_xmax, tolerance = 1e-4)
  expect_equal_to_r(result$y$boxes[1, 2], expected_ymin, tolerance = 1e-4)
  expect_equal_to_r(result$y$boxes[1, 4], expected_ymax, tolerance = 1e-4)
  expect_equal_to_r(result$y$boxes[1, 5], angle, tolerance = 1e-4)
})

test_that("item_transform_rotate converts boxes to xyxyr format", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$y$boxes$shape[2], 5)
  expect_equal_to_r(result$y$boxes[1, 5], 0)
})

test_that("item_transform_rotate applies angle to boxes", {
  item <- make_detection_item(
    matrix(c(100, 100, 102, 102), ncol = 4),
    image_size = c(200L, 200L)
  )
  result <- item_transform_rotate(item, angle = 45)

  expect_equal_to_r(result$y$boxes[1, 5], 45, tolerance = 1e-5)
})

test_that("item_transform_rotate preserves labels", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels
  )
  original_labels <- item$y$labels$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
})

test_that("item_transform_rotate handles empty boxes", {
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_rotate(item, angle = 45)

  expect_s3_class(result, "image_with_rotated_box")
  expect_equal(result$y$boxes$shape, c(0, 5))
  expect_equal(result$y$boxes$dtype, torch_float())
})

test_that("item_transform_rotate handles multiple boxes", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(400, 300))
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$y$boxes$shape, c(3, 5))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes)
  expect_equal_to_r(result$y$boxes[, 5], rep(0,3))
})

test_that("item_transform_rotate does not mutate input", {
  boxes <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  item <- make_detection_item(boxes)
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()
  original_class <- class(item)

  result <- item_transform_rotate(item, angle = 30)

  expect_true(item$x$eq(original_img)$all()$item())
  expect_equal_to_r(item$y$boxes, as.array(original_boxes$cpu()))
  expect_equal(class(item), original_class)
})

test_that("item_transform_rotate returns image_with_rotated_box class", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 45)

  expect_s3_class(result, "image_with_rotated_box")
  expect_false(inherits(result, "image_with_bounding_box"))
})

test_that("item_transform_rotate image dtype is preserved", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$x$dtype, item$x$dtype)
})

test_that("item_transform_rotate can be composed", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  labels <- sample.int(2^16, 3)
  item <- make_detection_item(boxes, labels = labels, image_size = c(410, 300))
  result <- item |>
    item_transform_rotate(angle = 90) |>
    item_transform_rotate(angle = 90)

  expect_equal(result$x$shape, c(3, 410, 300))
  expect_equal(result$y$image_height, 410)
  expect_equal(result$y$image_width, 300)
  expect_equal(result$y$boxes$shape, c(3, 5))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[, 5], rep(180, 3))
  expect_equal(result$y$labels, labels)
})

