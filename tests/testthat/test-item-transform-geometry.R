context("item-transform-geometry")

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

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 1:4], c(10, 20, 50, 60))
  expect_equal_to_r(result$y$boxes[1, 5], 0)
})

test_that("item_transform_rotate does not modify the image", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_x <- item$x$clone()
  result <- item_transform_rotate(item, angle = 90)

  expect_true(result$x$eq(original_x)$all()$item())
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 5], 90)
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

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(0, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_rotate handles multiple boxes with angle=0", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  item <- make_item(boxes)
  result <- item_transform_rotate(item, angle = 0)

  expect_tensor_shape(result$y$boxes, c(3, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[, 5], c(0, 0, 0))
})

test_that("item_transform_rotate does not mutate input", {
  boxes <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  item <- make_item(boxes)
  original_boxes <- boxes$clone()
  original_x <- item$x$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_equal_to_r(item$y$boxes, as.array(original_boxes$cpu()))
  expect_tensor_shape(item$y$boxes, c(1, 4))
  expect_false(inherits(item, "image_with_rotated_box"))
  expect_true(item$x$eq(original_x)$all()$item())
})

test_that("item_transform_rotate applies non-zero rotation angle to boxes", {
  item <- make_item(boxes = matrix(c(100, 100, 102, 102), ncol = 4), image_size = c(200L, 200L))
  result <- item_transform_rotate(item, angle = 45)

  cx <- 101; cy <- 101
  hw <- 1; hh <- 1
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 1], cx - hw, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 3], cx + hw, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 5], 45, tolerance = 1e-5)

  angle_rad <- 45 * pi / 180
  ct <- cos(angle_rad); st <- sin(angle_rad)
  corners_x <- c(cx - hw*ct + hh*st, cx + hw*ct + hh*st, cx + hw*ct - hh*st, cx - hw*ct - hh*st)
  corners_y <- c(cy - hw*st - hh*ct, cy + hw*st - hh*ct, cy + hw*st + hh*ct, cy - hw*st + hh*ct)
  expect_true(all(corners_x >= 0 & corners_x <= 200))
  expect_true(all(corners_y >= 0 & corners_y <= 200))
})
