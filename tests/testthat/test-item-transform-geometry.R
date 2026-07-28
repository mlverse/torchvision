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
  item <- make_item(boxes)
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_s3_class(result, "image_with_rotated_box")
  expect_true(result$x$eq(original_img)$all()$item())
  expect_equal(result$y$boxes$shape, c(1, 5))
  expect_equal_to_r(result$y$boxes[1, 5], 0)
})

test_that("item_transform_rotate expands canvas for non-axis-aligned angles", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(300L, 500L))
  result <- item_transform_rotate(item, angle = 30)

  new_W <- as.integer(ceiling(500 * abs(cos(30 * pi / 180)) + 300 * abs(sin(30 * pi / 180))))
  new_H <- as.integer(ceiling(500 * abs(sin(30 * pi / 180)) + 300 * abs(cos(30 * pi / 180))))

  expect_equal(result$x$shape[1], 3)
  expect_equal(result$x$shape[2], new_H)
  expect_equal(result$x$shape[3], new_W)
  expect_equal(result$y$image_height, new_H)
  expect_equal(result$y$image_width, new_W)
})

test_that("item_transform_rotate 90 degrees swaps dimensions", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_rotate(item, angle = 90)

  expect_equal(result$x$shape, c(3, 400, 200))
  expect_equal(result$y$image_height, 400L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_rotate 180 degrees preserves dimensions", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_rotate(item, angle = 180)

  expect_equal(result$x$shape, c(3, 200, 400))
  expect_equal(result$y$image_height, 200L)
  expect_equal(result$y$image_width, 400L)
})

test_that("item_transform_rotate negative angles work", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_rotate(item, angle = -45)

  new_W <- as.integer(ceiling(200 * abs(cos(-45 * pi / 180)) + 100 * abs(sin(-45 * pi / 180))))
  new_H <- as.integer(ceiling(200 * abs(sin(-45 * pi / 180)) + 100 * abs(cos(-45 * pi / 180))))

  expect_equal(result$x$shape[1], 3)
  expect_equal(result$x$shape[2], new_H)
  expect_equal(result$x$shape[3], new_W)
})

test_that("item_transform_rotate boxes are shifted for expanded canvas", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(300L, 500L))
  result <- item_transform_rotate(item, angle = 30)

  new_W <- as.integer(ceiling(500 * abs(cos(30 * pi / 180)) + 300 * abs(sin(30 * pi / 180))))
  new_H <- as.integer(ceiling(500 * abs(sin(30 * pi / 180)) + 300 * abs(cos(30 * pi / 180))))
  dx <- (new_W - 500) / 2
  dy <- (new_H - 300) / 2

  expect_equal_to_r(result$y$boxes[1, 1], 10 + dx, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 3], 50 + dx, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 2], 20 + dy, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 4], 60 + dy, tolerance = 1e-5)
})

test_that("item_transform_rotate converts boxes to xyxyr format", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$y$boxes$shape[2], 5)
  expect_equal_to_r(result$y$boxes[1, 5], 0)
})

test_that("item_transform_rotate applies angle to boxes", {
  item <- make_item(
    matrix(c(100, 100, 102, 102), ncol = 4),
    image_size = c(200L, 200L)
  )
  result <- item_transform_rotate(item, angle = 45)

  expect_equal_to_r(result$y$boxes[1, 5], 45, tolerance = 1e-5)
})

test_that("item_transform_rotate preserves labels", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels
  )
  original_labels <- item$y$labels$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
})

test_that("item_transform_rotate handles empty boxes", {
  item <- make_item(
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
  item <- make_item(boxes)
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$y$boxes$shape, c(3, 5))
  expect_equal_to_r(result$y$boxes[, 5], c(0, 0, 0))
})

test_that("item_transform_rotate does not mutate input", {
  boxes <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  item <- make_item(boxes)
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()
  original_class <- class(item)

  result <- item_transform_rotate(item, angle = 30)

  expect_true(item$x$eq(original_img)$all()$item())
  expect_equal_to_r(item$y$boxes, as.array(original_boxes$cpu()))
  expect_equal(class(item), original_class)
})

test_that("item_transform_rotate returns image_with_rotated_box class", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 45)

  expect_s3_class(result, "image_with_rotated_box")
  expect_false(inherits(result, "image_with_bounding_box"))
})

test_that("item_transform_rotate image dtype is preserved", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_rotate(item, angle = 0)

  expect_equal(result$x$dtype, item$x$dtype)
})


# --- item_transform_hflip tests ---

test_that("item_transform_hflip rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_hflip(img),
    "requires a dataset item"
  )
})

test_that("item_transform_hflip rejects numeric input", {
  expect_error(
    item_transform_hflip(42),
    "requires a dataset item"
  )
})

test_that("item_transform_hflip preserves image shape for detection", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$x, c(3, 100, 200))
})

test_that("item_transform_hflip flips box x-coordinates", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_equal_to_r(result$y$boxes[1, 1], 200 - 50)
  expect_equal_to_r(result$y$boxes[1, 3], 200 - 10)
})

test_that("item_transform_hflip preserves box y-coordinates", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_equal_to_r(result$y$boxes[1, 2], 20)
  expect_equal_to_r(result$y$boxes[1, 4], 60)
})

test_that("item_transform_hflip preserves labels for detection", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels
  )
  result <- item_transform_hflip(item)

  expect_true(result$y$labels$eq(labels)$all()$item())
})

test_that("item_transform_hflip preserves image_height and image_width for detection", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_hflip handles empty boxes", {
  item <- make_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_hflip handles multiple boxes", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  item <- make_item(boxes, image_size = c(500L, 600L))
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[1, 1], 600 - 50)
  expect_equal_to_r(result$y$boxes[1, 3], 600 - 10)
  expect_equal_to_r(result$y$boxes[2, 1], 600 - 150)
  expect_equal_to_r(result$y$boxes[2, 3], 600 - 100)
  expect_equal_to_r(result$y$boxes[3, 1], 600 - 300)
  expect_equal_to_r(result$y$boxes[3, 3], 600 - 0)
})

test_that("item_transform_hflip does not mutate input for detection", {
  boxes <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  item <- make_item(boxes)
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()
  original_class <- class(item)

  result <- item_transform_hflip(item)

  expect_true(item$x$eq(original_img)$all()$item())
  expect_equal_to_r(item$y$boxes, as.array(original_boxes$cpu()))
  expect_equal(class(item), original_class)
})

test_that("item_transform_hflip preserves class for detection", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_hflip(item)

  expect_s3_class(result, "image_with_bounding_box")
})

test_that("item_transform_hflip actually flips image pixels for detection", {
  w <- 200L
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, w))
  original_img <- item$x$clone()
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$x, c(3, 100L, w))
  expect_true(torch_equal(result$x, transform_hflip(original_img)))
})

test_that("item_transform_hflip image dtype is preserved for detection", {
  item <- make_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_hflip(item)

  expect_tensor_dtype(result$x, item$x$dtype)
})

test_that("item_transform_hflip preserves image shape for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$x, c(3, 100, 200))
})

test_that("item_transform_hflip flips masks for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_hflip(item)

  expect_tensor_shape(result$y$masks, original_masks$shape)
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(result$y$masks$equal(original_masks$flip(-1)))
})

test_that("item_transform_hflip preserves labels for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_labels <- item$y$labels$clone()

  result <- item_transform_hflip(item)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
})

test_that("item_transform_hflip preserves image_height and image_width for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_hflip preserves class for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_s3_class(result, "image_with_segmentation_mask")
})

test_that("item_transform_hflip image dtype is preserved for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_tensor_dtype(result$x, item$x$dtype)
})
