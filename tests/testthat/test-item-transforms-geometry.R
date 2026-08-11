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
  expect_tensor_shape(result$y$boxes, c(1, 5))
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
  expect_tensor_shape(result$x, c(3, 400L, 200L))
  expect_equal(result$y$image_height, 400L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_rotate 180 degrees preserves dimensions", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_rotate(item, angle = 180)

  expect_tensor_shape(result$x, c(3, 200, 400))
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

  # Compute geometry manually
  orig_box <- c(10, 20, 50, 60)
  angle <- 30

  cx <- (orig_box[1] + orig_box[3]) / 2  # 30
  cy <- (orig_box[2] + orig_box[4]) / 2  # 40
  hw <- (orig_box[3] - orig_box[1]) / 2  # 20
  hh <- (orig_box[4] - orig_box[2]) / 2  # 20

  angle_rad <- angle * pi / 180
  new_hw <- hw * abs(cos(angle_rad)) + hh * abs(sin(angle_rad)) # ~27.32
  new_hh <- hw * abs(sin(angle_rad)) + hh * abs(cos(angle_rad)) # ~27.32

  new_H <- as.integer(ceiling(500 * abs(sin(angle_rad)) + 300 * abs(cos(angle_rad))))
  new_W <- as.integer(ceiling(500 * abs(cos(angle_rad)) + 300 * abs(sin(angle_rad))))
  dx <- (new_W - 500) / 2
  dy <- (new_H - 300) / 2

  expected_xmin <- cx + dx - new_hw
  expected_xmax <- cx + dx + new_hw
  expected_ymin <- cy + dy - new_hh
  expected_ymax <- cy + dy + new_hh

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
  expect_tensor_shape(result$y$boxes, c(0, 5))
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

  expect_tensor_shape(result$y$boxes, c(3, 5))
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

  expect_tensor_shape(result$x, c(3, 410, 300))
  expect_equal(result$y$image_height, 410)
  expect_equal(result$y$image_width, 300)
  expect_tensor_shape(result$y$boxes, c(3, 5))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[, 5], rep(180, 3))
  expect_equal(result$y$labels, labels)
})

test_that("item_transform_rotate 0 degrees preserves masks for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_rotate(item, angle = 0)

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_true(result$y$masks$equal(original_masks))
})

test_that("item_transform_rotate rotates masks for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_rotate(item, angle = 90)

  expect_tensor_shape(result$y$masks, c(2, 200, 100))
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(result$y$masks$equal(original_masks$transpose(-1, -2)$flip(-2)))
})

test_that("item_transform_rotate expands image and masks together for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  result <- item_transform_rotate(item, angle = 30)

  expect_equal(result$y$masks$shape[2], result$x$shape[2])
  expect_equal(result$y$masks$shape[3], result$x$shape[3])
  expect_equal(result$y$image_height, result$x$shape[2])
  expect_equal(result$y$image_width, result$x$shape[3])
})

test_that("item_transform_rotate preserves labels for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_labels <- item$y$labels$clone()

  result <- item_transform_rotate(item, angle = 30)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
})

test_that("item_transform_rotate works on a segmentation dataset", {
  ds <- dataset(
    name = "toy_segmentation",
    initialize = function() {},
    .getitem = function(index) {
      make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
    },
    .length = function() 1L
  )()

  ds <- item_transform_rotate(ds, angle = 90)
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_segmentation_mask")
  expect_tensor_shape(item$x, c(3, 200, 100))
  expect_tensor_shape(item$y$masks, c(2, 200, 100))
})

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

test_that("item_transform_hflip preserves image shape and flips x-coordinates", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal_to_r(result$y$boxes[1, 1], 200 - 50)
  expect_equal_to_r(result$y$boxes[1, 3], 200 - 10)
  expect_equal_to_r(result$y$boxes[1, 2], 20)
  expect_equal_to_r(result$y$boxes[1, 4], 60)
})

test_that("item_transform_hflip preserves labels and metadata", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 200L)
  )
  result <- item_transform_hflip(item)

  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_hflip handles empty boxes", {
  item <- make_detection_item(
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
  item <- make_detection_item(boxes, image_size = c(500L, 600L))
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[1, 1], 600 - 50)
  expect_equal_to_r(result$y$boxes[1, 3], 600 - 10)
  expect_equal_to_r(result$y$boxes[2, 1], 600 - 150)
  expect_equal_to_r(result$y$boxes[2, 3], 600 - 100)
  expect_equal_to_r(result$y$boxes[3, 1], 600 - 300)
  expect_equal_to_r(result$y$boxes[3, 3], 600 - 0)
  expect_equal_to_r(result$y$boxes[1, 2], 20)
  expect_equal_to_r(result$y$boxes[1, 4], 60)
  expect_equal_to_r(result$y$boxes[2, 2], 200)
  expect_equal_to_r(result$y$boxes[2, 4], 250)
  expect_equal_to_r(result$y$boxes[3, 2], 0)
  expect_equal_to_r(result$y$boxes[3, 4], 400)
})

test_that("item_transform_hflip does not mutate input for detection", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(torch_tensor(boxes))
  original_img <- as_array(item$x)
  original_class <- class(item)

  result <- item_transform_hflip(item)

  expect_equal_to_r(item$x, original_img)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)
})

test_that("item_transform_hflip preserves class", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_hflip(item)

  expect_s3_class(result, "image_with_bounding_box")
})

test_that("item_transform_hflip actually flips image pixels", {
  w <- 200L
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, w))
  original_img <- item$x$clone()
  result <- item_transform_hflip(item)

  expect_tensor_shape(result$x, c(3, 100L, w))
  expect_true(torch_equal(result$x, transform_hflip(original_img)))
})

test_that("item_transform_hflip image dtype is preserved for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
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
  original_labels <- as.integer(as_array(item$y$labels))

  result <- item_transform_hflip(item)

  expect_equal_to_r(result$y$labels, original_labels)
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

test_that("item_transform_hflip can be composed", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  labels <- sample.int(2^16, 3)
  item <- make_detection_item(boxes, labels = labels, image_size = c(410, 300))
  result <- item |>
    item_transform_hflip() |>
    item_transform_hflip()

  expect_tensor_shape(result$x, c(3, 410, 300))
  expect_equal(result$y$image_height, 410)
  expect_equal(result$y$image_width, 300)
  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes, tolerance = 1e-5)
  expect_equal(result$y$labels, labels)
})

test_that("item_transform_hflip handles rotated boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_hflip(rotated)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 5], -30, tolerance = 1e-5)
})

test_that("item_transform_hflip composed with rotate is symmetric", {
  boxes <- matrix(c(10, 20, 50, 60, 100, 200, 150, 250), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(300L, 400L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- rotated |>
    item_transform_hflip() |>
    item_transform_hflip()

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(2, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes, as_array(rotated$y$boxes), tolerance = 1e-5)
})

# item_transform_vflip

test_that("item_transform_vflip rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_vflip(img),
    "requires a dataset item"
  )
})

test_that("item_transform_vflip rejects numeric input", {
  expect_error(
    item_transform_vflip(42),
    "requires a dataset item"
  )
})

test_that("item_transform_vflip preserves image shape and flips y-coordinates", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_vflip(item)

  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal_to_r(result$y$boxes[1, 1], 10)
  expect_equal_to_r(result$y$boxes[1, 3], 50)
  expect_equal_to_r(result$y$boxes[1, 2], 100 - 60)
  expect_equal_to_r(result$y$boxes[1, 4], 100 - 20)
})

test_that("item_transform_vflip preserves labels and metadata", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 200L)
  )
  result <- item_transform_vflip(item)

  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_vflip handles empty boxes", {
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_vflip(item)

  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_vflip handles multiple boxes", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(500L, 600L))
  result <- item_transform_vflip(item)

  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[1, 1], 10)
  expect_equal_to_r(result$y$boxes[1, 3], 50)
  expect_equal_to_r(result$y$boxes[2, 1], 100)
  expect_equal_to_r(result$y$boxes[2, 3], 150)
  expect_equal_to_r(result$y$boxes[3, 1], 0)
  expect_equal_to_r(result$y$boxes[3, 3], 300)
  expect_equal_to_r(result$y$boxes[1, 2], 500 - 60)
  expect_equal_to_r(result$y$boxes[1, 4], 500 - 20)
  expect_equal_to_r(result$y$boxes[2, 2], 500 - 250)
  expect_equal_to_r(result$y$boxes[2, 4], 500 - 200)
  expect_equal_to_r(result$y$boxes[3, 2], 500 - 400)
  expect_equal_to_r(result$y$boxes[3, 4], 500 - 0)
})

test_that("item_transform_vflip does not mutate input for detection", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(torch_tensor(boxes))
  original_img <- as_array(item$x)
  original_class <- class(item)

  result <- item_transform_vflip(item)

  expect_equal_to_r(item$x, original_img)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)
})

test_that("item_transform_vflip preserves class", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_vflip(item)

  expect_s3_class(result, "image_with_bounding_box")
})

test_that("item_transform_vflip actually flips image pixels", {
  h <- 100L
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(h, 200L))
  original_img <- item$x$clone()
  result <- item_transform_vflip(item)

  expect_tensor_shape(result$x, c(3, h, 200L))
  expect_true(torch_equal(result$x, transform_vflip(original_img)))
})

test_that("item_transform_vflip image dtype is preserved for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_vflip(item)

  expect_tensor_dtype(result$x, item$x$dtype)
})

test_that("item_transform_vflip preserves image shape for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_vflip(item)

  expect_tensor_shape(result$x, c(3, 100, 200))
})

test_that("item_transform_vflip flips masks for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_vflip(item)

  expect_tensor_shape(result$y$masks, original_masks$shape)
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(result$y$masks$equal(original_masks$flip(-2)))
})

test_that("item_transform_vflip preserves labels for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_labels <- as.integer(as_array(item$y$labels))

  result <- item_transform_vflip(item)

  expect_equal_to_r(result$y$labels, original_labels)
})

test_that("item_transform_vflip preserves image_height and image_width for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_vflip(item)

  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_vflip preserves class for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_vflip(item)

  expect_s3_class(result, "image_with_segmentation_mask")
})

test_that("item_transform_vflip image dtype is preserved for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_vflip(item)

  expect_tensor_dtype(result$x, item$x$dtype)
})

test_that("item_transform_vflip can be composed", {
  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  labels <- sample.int(2^16, 3)
  item <- make_detection_item(boxes, labels = labels, image_size = c(410, 300))
  result <- item |>
    item_transform_vflip() |>
    item_transform_vflip()

  expect_tensor_shape(result$x, c(3, 410, 300))
  expect_equal(result$y$image_height, 410)
  expect_equal(result$y$image_width, 300)
  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[, 1:4], boxes, tolerance = 1e-5)
  expect_equal(result$y$labels, labels)
})

test_that("item_transform_vflip handles rotated boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_vflip(rotated)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 5], -30, tolerance = 1e-5)
})

test_that("item_transform_vflip composed with rotate is symmetric", {
  boxes <- matrix(c(10, 20, 50, 60, 100, 200, 150, 250), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(300L, 400L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- rotated |>
    item_transform_vflip() |>
    item_transform_vflip()

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(2, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes, as_array(rotated$y$boxes), tolerance = 1e-5)
})

# --- item_transform_center_crop tests ---

test_that("item_transform_center_crop rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_center_crop(img, size = 50),
    "requires a dataset item"
  )
  expect_error(
    item_transform_center_crop(42, size = 50),
    "requires a dataset item"
  )
})

test_that("item_transform_center_crop crops detection items and adjusts targets", {
  boxes <- matrix(c(120, 70, 180, 130), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(200L, 400L))
  original_img <- item$x$clone()
  original_img_r <- as_array(item$x)
  original_class <- class(item)
  original_dtype <- item$x$dtype

  result <- item_transform_center_crop(item, size = c(100L, 200L))

  expect_s3_class(result, "image_with_bounding_box")
  expect_tensor_dtype(result$x, original_dtype)
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
  expect_true(torch_equal(result$x, transform_center_crop(original_img, size = c(100L, 200L))))
  expect_equal_to_r(result$y$boxes[1, 1], 120 - 99) # x1
  expect_equal_to_r(result$y$boxes[1, 3], 180 - 99) # x2
  expect_equal_to_r(result$y$boxes[1, 2], 70 - 49)  # y1
  expect_equal_to_r(result$y$boxes[1, 4], 130 - 49) # y2

  # input is not mutated
  expect_equal_to_r(item$x, original_img_r)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)

  # labels and metadata are preserved
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(120, 70, 180, 130, 210, 80, 280, 120), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(200L, 400L)
  )
  result <- item_transform_center_crop(item, size = c(100L, 200L))

  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)

  # multiple boxes are adjusted and clamped to the crop
  boxes <- matrix(c(
    120, 70,  180, 130,
    150, 60,  280, 140,
    110, 80,  140, 120
  ), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(200L, 400L))
  result <- item_transform_center_crop(item, size = c(100L, 200L))

  expected_boxes <- boxes
  expected_boxes[, 1] <- pmax(0, boxes[, 1] - 99)
  expected_boxes[, 3] <- pmin(200, boxes[, 3] - 99)
  expected_boxes[, 2] <- pmax(0, boxes[, 2] - 49)
  expected_boxes[, 4] <- pmin(100, boxes[, 4] - 49)
  expect_tensor_shape(result$y$boxes, c(3, 4))
  expect_equal_to_r(result$y$boxes[, 1], expected_boxes[, 1])
  expect_equal_to_r(result$y$boxes[, 3], expected_boxes[, 3])
  expect_equal_to_r(result$y$boxes[, 2], expected_boxes[, 2])
  expect_equal_to_r(result$y$boxes[, 4], expected_boxes[, 4])

  # empty boxes are preserved
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_center_crop(item, size = c(100L, 200L))

  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_center_crop square crop via single int", {
  item <- make_detection_item(matrix(c(120, 70, 180, 130), ncol = 4), image_size = c(200L, 400L))
  result <- item_transform_center_crop(item, size = 100)

  expect_tensor_shape(result$x, c(3, 100, 100))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 100L)
})

test_that("item_transform_center_crop crops segmentation items", {
  item <- make_segmentation_item(image_size = c(200L, 400L), num_masks = 2L)
  original_img <- item$x$clone()
  original_masks <- item$y$masks$clone()
  original_dtype <- item$x$dtype
  original_labels <- as.integer(as_array(item$y$labels))

  result <- item_transform_center_crop(item, size = c(100L, 200L))

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_tensor_dtype(result$x, original_dtype)
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
  expect_equal_to_r(result$y$labels, original_labels)

  expected_masks <- transform_center_crop(original_masks, size = c(100L, 200L))
  expect_tensor_shape(result$y$masks, c(2, 100, 200))
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(result$y$masks$equal(expected_masks))
})

test_that("item_transform_center_crop pads when crop is larger than image", {
  item <- make_detection_item(matrix(c(5, 5, 15, 15), ncol = 4), image_size = c(20L, 30L))
  result <- item_transform_center_crop(item, size = c(30L, 40L))

  expect_tensor_shape(result$x, c(3, 30, 40))
  expect_equal(result$y$image_height, 30L)
  expect_equal(result$y$image_width, 40L)
})

test_that("item_transform_center_crop can be composed", {
  boxes <- matrix(c(120, 70, 180, 130, 210, 80, 280, 120), ncol = 4, byrow = TRUE)
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(boxes, labels = labels, image_size = c(200L, 400L))

  result <- item |>
    item_transform_hflip() |>
    item_transform_center_crop(size = c(100L, 200L))

  # after hflip, boxes become (W - x2, y1, W - x1, y2) with W = 400;
  # center crop offsets are 99 (width) and 49 (height)
  expect_s3_class(result, "image_with_bounding_box")
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_equal_to_r(result$y$boxes[1, ], c(220 - 99, 70 - 49, 280 - 99, 130 - 49))
  expect_equal_to_r(result$y$boxes[2, ], c(120 - 99, 80 - 49, 190 - 99, 120 - 49))
})

test_that("item_transform_center_crop handles rotated boxes", {
  boxes <- matrix(c(120, 70, 180, 130), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(200L, 400L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_center_crop(rotated, size = c(100L, 200L))

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
})



# item_transform_affine

test_that("item_transform_affine rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_affine(img, angle = 30),
    "requires a dataset item"
  )
})

test_that("item_transform_affine rejects numeric input", {
  expect_error(
    item_transform_affine(42),
    "requires a dataset item"
  )
})

test_that("item_transform_affine identity preserves image and boxes", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_img <- item$x$clone()

  result <- item_transform_affine(item)

  expect_s3_class(result, "image_with_rotated_box")
  expect_true(torch_equal(result$x, original_img))
  expect_equal_to_r(result$y$boxes, matrix(c(10, 20, 50, 60, 0), ncol = 5))
})

test_that("item_transform_affine returns rotated boxes in xyxyr format", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_affine(item, angle = 25)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_equal_to_r(result$y$boxes[1, 5], 25)
})

test_that("item_transform_affine preserves image shape for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_affine(item, angle = 25, translate = c(10, 5),
                                  scale = 1.2, shear = 8)

  expect_tensor_shape(result$x, c(3, 100, 200))
})

test_that("item_transform_affine transforms the image like transform_affine", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_img <- item$x$clone()

  result <- item_transform_affine(item, angle = 15, translate = c(4, -3),
                                  scale = 0.9, shear = 5)
  expected <- transform_affine(original_img, angle = 15, translate = c(4, -3),
                               scale = 0.9, shear = 5)

  expect_true(torch_equal(result$x, expected))
})

test_that("item_transform_affine translates box centre", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_affine(item, translate = c(30, 10))

  expect_equal_to_r(result$y$boxes, matrix(c(40, 30, 80, 70, 0), ncol = 5))
})

test_that("item_transform_affine scales boxes around the image center", {
  item <- make_detection_item(matrix(c(80, 40, 120, 60), ncol = 4), image_size = c(100L, 200L))
  result <- item_transform_affine(item, scale = 2)

  expect_equal_to_r(result$y$boxes, matrix(c(60, 30, 140, 70, 0), ncol = 5))
})

test_that("item_transform_affine rotates boxes around the image center", {
  item <- make_detection_item(matrix(c(50, 80, 150, 120), ncol = 4), image_size = c(200L, 200L))
  result <- item_transform_affine(item, angle = 90)

  expect_equal(as.numeric(result$y$boxes[1, 1]), 80, tolerance = 1e-4)
  expect_equal(as.numeric(result$y$boxes[1, 2]), 50, tolerance = 1e-4)
  expect_equal(as.numeric(result$y$boxes[1, 3]), 120, tolerance = 1e-4)
  expect_equal(as.numeric(result$y$boxes[1, 4]), 150, tolerance = 1e-4)
  expect_equal(as.numeric(result$y$boxes[1, 5]), 90, tolerance = 1e-4)
})

test_that("item_transform_affine handles empty boxes", {
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_affine(item, angle = 30)

  expect_tensor_shape(result$y$boxes, c(0, 5))
})

test_that("item_transform_affine preserves labels for detection", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels
  )
  result <- item_transform_affine(item, angle = 20)

  expect_true(result$y$labels$eq(labels)$all()$item())
})

test_that("item_transform_affine does not mutate input for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()

  item_transform_affine(item, angle = 30, translate = c(10, 10))

  expect_true(torch_equal(item$x, original_img))
  expect_equal_to_r(item$y$boxes, as.array(original_boxes$cpu()))
})

test_that("item_transform_affine image dtype is preserved for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_affine(item, angle = 10)

  expect_tensor_dtype(result$x, item$x$dtype)
})

test_that("item_transform_affine can be composed", {
  item <- make_detection_item(matrix(c(50, 80, 150, 120), ncol = 4), image_size = c(200L, 200L))
  composed <- item |>
    item_transform_affine(angle = 30) |>
    item_transform_affine(angle = 30)
  single <- item_transform_affine(item, angle = 60)

  expect_s3_class(composed, "image_with_rotated_box")
  expect_equal_to_r(composed$y$boxes, as.array(single$y$boxes$cpu()), tolerance = 1e-4)
})

test_that("item_transform_affine composed on empty boxes stays xyxyr", {
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  composed <- item |>
    item_transform_affine(angle = 30) |>
    item_transform_affine(angle = 30)

  expect_tensor_shape(composed$y$boxes, c(0, 5))
})

test_that("item_transform_affine preserves image shape and class for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L))
  result <- item_transform_affine(item, angle = 20, translate = c(5, 5), scale = 1.1)

  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_s3_class(result, "image_with_segmentation_mask")
})

test_that("item_transform_affine identity preserves masks", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_affine(item)

  expect_true(result$y$masks$equal(original_masks))
})

test_that("item_transform_affine transforms masks and preserves dtype", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_affine(item, angle = 45)

  expect_tensor_shape(result$y$masks, original_masks$shape)
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_false(result$y$masks$equal(original_masks))
})

test_that("item_transform_affine preserves labels for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_labels <- item$y$labels$clone()

  result <- item_transform_affine(item, angle = 20)

  expect_true(result$y$labels$eq(original_labels)$all()$item())
})

test_that("item_transform_affine works on a dataset", {
  ds <- dataset(
    name = "toy_detection",
    initialize = function() {},
    .getitem = function(index) {
      make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
    },
    .length = function() 1L
  )()

  ds <- item_transform_affine(ds, translate = c(30, 10))
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_rotated_box")
  expect_equal_to_r(item$y$boxes, matrix(c(40, 30, 80, 70, 0), ncol = 5))
})

# item_transform_crop

  test_that("item_transform_crop rejects non-item inputs", {
    img <- torch_randn(3, 100, 200)
    expect_error(
      item_transform_crop(img, top = 1, left = 1, height = 50, width = 100),
      "requires a dataset item"
    )
    expect_error(
      item_transform_crop(42, top = 1, left = 1, height = 50, width = 100),
      "requires a dataset item"
    )
  })

test_that("item_transform_crop crops detection items correctly", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()
  result <- item_transform_crop(item, top = 11, left = 21, height = 80, width = 160)

  expect_tensor_shape(result$x, c(3, 80, 160))
  expect_equal(result$y$image_height, 80L)
  expect_equal(result$y$image_width, 160L)
  expect_true(torch_equal(result$x, transform_crop(original_img, top = 11, left = 21, height = 80, width = 160)))

  # offset_x = left - 1 = 20, offset_y = top - 1 = 10
  # new_x1 = max(0, 10 - 20) = 0, new_x2 = min(160, 50 - 20) = 30
  # new_y1 = max(0, 20 - 10) = 10, new_y2 = min(80, 60 - 10) = 50
  expect_equal_to_r(result$y$boxes[1, 1], 0)
  expect_equal_to_r(result$y$boxes[1, 3], 30)
  expect_equal_to_r(result$y$boxes[1, 2], 10)
  expect_equal_to_r(result$y$boxes[1, 4], 50)
})

test_that("item_transform_crop removes boxes outside crop area", {
  # Box at (150, 50, 180, 80) — intersects the crop (left = 140, width = 40, offset 139)
  item <- make_detection_item(
    matrix(c(150, 50, 180, 80), ncol = 4),
    image_size = c(200L, 300L)
  )
  result <- item_transform_crop(item, top = 1, left = 140, height = 100, width = 40)

  # offset_x = 139, new_x1 = 150 - 139 = 11, new_x2 = 180 - 139 = 41
  # clipped: new_x1 = max(0, 11) = 11, new_x2 = min(40, 41) = 40
  # keep = (40 > 11) & ... = TRUE
  expect_tensor_shape(result$y$boxes, c(1, 4))

  # Box at (200, 10, 250, 50) — entirely outside crop left=1, width=100 (x range [0, 100))
  # offset_x = 0, new_x1 = 200, new_x2 = 250, clipped: x1=100, x2=100 → zero width
  item_outside <- make_detection_item(
    matrix(c(200, 10, 250, 50), ncol = 4),
    image_size = c(200L, 300L)
  )
  result_outside <- item_transform_crop(item_outside, top = 1, left = 1, height = 100, width = 100)

  expect_tensor_shape(result_outside$y$boxes, c(0, 4))
})

test_that("item_transform_crop handles multiple boxes and preserves labels", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 100, 20, 180, 80), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 200L)
  )
  result <- item_transform_crop(item, top = 11, left = 21, height = 80, width = 160)

  expect_tensor_shape(result$y$boxes, c(2, 4))
  expect_equal(result$y$labels$size(1), result$y$boxes$size(1))
})

test_that("item_transform_crop handles empty boxes", {
  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_crop(item, top = 1, left = 1, height = 50, width = 100)

  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_crop does not mutate input", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(torch_tensor(boxes))
  original_img <- as_array(item$x)
  original_class <- class(item)

  result <- item_transform_crop(item, top = 6, left = 6, height = 50, width = 100)

  expect_equal_to_r(item$x, original_img)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)
})

test_that("item_transform_crop preserves class and dtype for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- item_transform_crop(item, top = 1, left = 1, height = 50, width = 100)

  expect_s3_class(result, "image_with_bounding_box")
  expect_tensor_dtype(result$x, item$x$dtype)
})

test_that("item_transform_crop transforms segmentation items", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()
  original_labels <- as.integer(as_array(item$y$labels))
  original_dtype <- item$x$dtype

  result <- item_transform_crop(item, top = 11, left = 21, height = 80, width = 160)

  expected_masks <- transform_crop(original_masks, top = 11, left = 21, height = 80, width = 160)

  expect_tensor_shape(result$x, c(3, 80, 160))
  expect_tensor_dtype(result$x, original_dtype)
  expect_tensor_shape(result$y$masks, c(2, 80, 160))
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(result$y$masks$equal(expected_masks))
  expect_equal_to_r(result$y$labels, original_labels)
  expect_equal(result$y$image_height, 80L)
  expect_equal(result$y$image_width, 160L)
  expect_s3_class(result, "image_with_segmentation_mask")
})

test_that("item_transform_crop transforms rotated-box items", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)
  original_angles <- as_array(rotated$y$boxes[, 5])

  result <- item_transform_crop(rotated, top = 11, left = 21, height = 80, width = 160)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$x, c(3, 80, 160))
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal(result$y$image_height, 80L)
  expect_equal(result$y$image_width, 160L)
  expect_equal_to_r(result$y$boxes[, 5], original_angles)
})

test_that("item_transform_crop removes rotated boxes outside crop area", {
  item <- make_detection_item(
    matrix(c(150, 50, 180, 80), ncol = 4),
    image_size = c(200L, 300L)
  )
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_crop(rotated, top = 1, left = 140, height = 100, width = 40)

  expect_tensor_shape(result$y$boxes, c(0, 5))
})

#  item_transform_pad

test_that("item_transform_pad rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_pad(img, padding = 10),
    "requires a dataset item"
  )
  expect_error(
    item_transform_pad(42, padding = 10),
    "requires a dataset item"
  )
})

test_that("item_transform_pad supports single-int, length-2, and length-4 padding", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  single <- item_transform_pad(item, padding = 10)
  expect_tensor_shape(single$x, c(3, 120, 220))
  expect_true(torch_equal(single$x, transform_pad(item$x, 10)))
  expect_equal_to_r(single$y$boxes[1, ], c(20, 30, 60, 70))
  expect_equal(single$y$image_height, 120L)
  expect_equal(single$y$image_width, 220L)

  len2 <- item_transform_pad(item, padding = c(5, 15))
  expect_tensor_shape(len2$x, c(3, 130, 210))
  expect_equal_to_r(len2$y$boxes[1, 1], 15)
  expect_equal_to_r(len2$y$boxes[1, 2], 35)
  expect_equal(len2$y$image_height, 130L)
  expect_equal(len2$y$image_width, 210L)

  len4 <- item_transform_pad(item, padding = c(5, 7, 9, 11))
  expect_tensor_shape(len4$x, c(3, 120, 212))
  expect_equal_to_r(len4$y$boxes[1, ], c(15, 29, 55, 69))
  expect_equal(len4$y$image_height, 120L)
  expect_equal(len4$y$image_width, 212L)
})

test_that("item_transform_pad preserves labels and metadata", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 200L)
  )
  result <- item_transform_pad(item, padding = 5)

  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_tensor_shape(result$y$area, c(2))
  expect_tensor_shape(result$y$iscrowd, c(2))
  expect_equal(result$y$image_height, 110L)
  expect_equal(result$y$image_width, 210L)
})

test_that("item_transform_pad handles box edge cases", {
  empty <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result_empty <- item_transform_pad(empty, padding = 5)
  expect_tensor_shape(result_empty$y$boxes, c(0, 4))
  expect_tensor_dtype(result_empty$y$boxes, torch_float())

  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  multi <- make_detection_item(boxes, image_size = c(500L, 600L))
  result_multi <- item_transform_pad(multi, padding = 10)

  expect_tensor_shape(result_multi$y$boxes, c(3, 4))
  expect_equal_to_r(result_multi$y$boxes[1, 1], 20)
  expect_equal_to_r(result_multi$y$boxes[1, 3], 60)
  expect_equal_to_r(result_multi$y$boxes[2, 1], 110)
  expect_equal_to_r(result_multi$y$boxes[2, 3], 160)
  expect_equal_to_r(result_multi$y$boxes[3, 1], 10)
  expect_equal_to_r(result_multi$y$boxes[3, 3], 310)
})

test_that("item_transform_pad fills pixels without mutating input", {
  boxes <- matrix(c(2, 2, 4, 4), ncol = 4)
  item <- make_detection_item(torch_tensor(boxes), image_size = c(10L, 10L))
  item$x <- torch_ones(3, 10, 10)
  original_img <- as_array(item$x)
  original_class <- class(item)

  result <- item_transform_pad(item, padding = 2)
  expect_equal(as.numeric(result$x[1, 1, 1]$cpu()), 0)
  expect_equal(as.numeric(result$x[1, 3, 3]$cpu()), 1)
  expect_equal(as.numeric(result$x[1, 2, 11]$cpu()), 0)

  result_filled <- item_transform_pad(item, padding = 2, fill = 1)
  expect_equal(as.numeric(result_filled$x[1, 1, 1]$cpu()), 1)

  expect_equal_to_r(item$x, original_img)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)
  expect_equal(class(result), original_class)

  item$x <- torch_randn(3, 100, 200, dtype = torch_float64())
  result_dtype <- item_transform_pad(item, padding = 5)
  expect_tensor_dtype(result_dtype$x, torch_float64())
})

test_that("item_transform_pad pads masks for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()
  original_labels <- as.integer(as_array(item$y$labels))

  result <- item_transform_pad(item, padding = 5)

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_tensor_shape(result$x, c(3, 110, 210))
  expect_tensor_shape(result$y$masks, c(2, 110, 210))
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_true(torch_equal(result$y$masks, transform_pad(original_masks, 5, fill = 0)))
  expect_equal_to_r(result$y$labels, original_labels)
  expect_equal(result$y$image_height, 110L)
  expect_equal(result$y$image_width, 210L)
})

test_that("item_transform_pad handles rotated boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)
  original_boxes <- rotated$y$boxes$clone()

  result <- item_transform_pad(rotated, padding = 5)

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_equal_to_r(result$y$boxes[1, 1], as_array(original_boxes[1, 1]) + 5)
  expect_equal_to_r(result$y$boxes[1, 2], as_array(original_boxes[1, 2]) + 5)
  expect_equal_to_r(result$y$boxes[1, 3], as_array(original_boxes[1, 3]) + 5)
  expect_equal_to_r(result$y$boxes[1, 4], as_array(original_boxes[1, 4]) + 5)
  expect_equal_to_r(result$y$boxes[1, 5], 30, tolerance = 1e-5)
})

test_that("item_transform_pad negative padding clips and drops boxes outside the crop", {
  item <- make_detection_item(
    matrix(c(0, 0, 10, 10, 120, 70, 180, 130), ncol = 4, byrow = TRUE),
    labels = torch_tensor(c(1L, 2L), dtype = torch_long()),
    image_size = c(200L, 400L)
  )

  result <- item_transform_pad(item, padding = c(-50, 0, 0, 0))

  expect_equal_to_r(result$y$boxes, matrix(c(70, 70, 130, 130), ncol = 4))
  expect_equal_to_r(result$y$labels, 2L)
  expect_equal(result$y$image_height, 200L)
  expect_equal(result$y$image_width, 350L)
})

# --- item_transform_perspective tests ---

test_that("item_transform_perspective rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))
  ep <- sp
  expect_error(
    item_transform_perspective(img, sp, ep),
    "requires a dataset item"
  )
  expect_error(
    item_transform_perspective(42, sp, ep),
    "requires a dataset item"
  )
})

test_that("item_transform_perspective applies box transforms without mutating input", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(torch_tensor(boxes), image_size = c(100L, 200L))
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))
  original_img <- as_array(item$x)
  original_class <- class(item)

  identity <- item_transform_perspective(item, sp, sp)

  expect_s3_class(identity, "image_with_bounding_box")
  expect_tensor_shape(identity$x, c(3, 100, 200))
  expect_true(identity$x$allclose(item$x, atol = 1e-4))
  expect_equal_to_r(identity$y$boxes, boxes)
  expect_equal(identity$x$dtype, item$x$dtype)

  ep <- list(c(10, 20), c(209, 20), c(209, 119), c(10, 119))
  translated <- item_transform_perspective(item, sp, ep)

  expect_equal_to_r(translated$y$boxes[1, 1], 20)
  expect_equal_to_r(translated$y$boxes[1, 2], 40)
  expect_equal_to_r(translated$y$boxes[1, 3], 60)
  expect_equal_to_r(translated$y$boxes[1, 4], 80)

  expect_equal_to_r(item$x, original_img)
  expect_equal_to_r(item$y$boxes, boxes)
  expect_equal(class(item), original_class)
})

test_that("item_transform_perspective preserves labels and metadata", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 100, 30, 150, 80), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 200L)
  )
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))

  result <- item_transform_perspective(item, sp, sp)

  expect_equal_to_r(result$y$labels, as.integer(as_array(labels)))
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_perspective handles box edge cases", {
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))

  empty <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result_empty <- item_transform_perspective(empty, sp, sp)
  expect_tensor_shape(result_empty$y$boxes, c(0, 4))
  expect_tensor_dtype(result_empty$y$boxes, torch_float())

  boxes <- matrix(c(
    10, 20, 50, 60,
    100, 200, 150, 250,
    0, 0, 300, 400
  ), ncol = 4, byrow = TRUE)
  multi <- make_detection_item(boxes, image_size = c(500L, 600L))
  sp600 <- list(c(0, 0), c(599, 0), c(599, 499), c(0, 499))
  ep600 <- list(c(10, 20), c(609, 20), c(609, 519), c(10, 519))
  result_multi <- item_transform_perspective(multi, sp600, ep600)

  expect_tensor_shape(result_multi$y$boxes, c(3, 4))
  expect_equal_to_r(result_multi$y$boxes[, 1], boxes[, 1] + 10)
  expect_equal_to_r(result_multi$y$boxes[, 2], boxes[, 2] + 20)
  expect_equal_to_r(result_multi$y$boxes[, 3], pmin(600, boxes[, 3] + 10))
  expect_equal_to_r(result_multi$y$boxes[, 4], pmin(500, boxes[, 4] + 20))

  outside <- make_detection_item(matrix(c(80, 80, 90, 90), ncol = 4), image_size = c(100L, 100L))
  sp100 <- list(c(0, 0), c(99, 0), c(99, 99), c(0, 99))
  ep100 <- list(c(50, 0), c(149, 0), c(149, 99), c(50, 99))
  result_outside <- item_transform_perspective(outside, sp100, ep100)

  expect_tensor_shape(result_outside$y$boxes, c(0, 4))
  expect_tensor_shape(result_outside$y$labels, 0L)
})

test_that("item_transform_perspective transforms segmentation masks", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))
  ep <- list(c(10, 5), c(190, 0), c(195, 95), c(5, 100))
  original_masks <- item$y$masks$clone()
  original_dtype <- item$x$dtype
  original_labels <- as.integer(as_array(item$y$labels))

  result <- item_transform_perspective(item, sp, ep)

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_tensor_dtype(result$x, original_dtype)
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_tensor_shape(result$y$masks, original_masks$shape)
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_equal_to_r(result$y$labels, original_labels)
  expect_equal(result$y$image_height, 100L)
  expect_equal(result$y$image_width, 200L)

  expected_masks <- transform_perspective(original_masks, startpoints = sp, endpoints = ep,
                                          interpolation = 0)
  expect_true(result$y$masks$equal(expected_masks))
})

test_that("item_transform_perspective can be composed", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))
  ep <- list(c(10, 0), c(209, 0), c(209, 99), c(10, 99))

  result <- item |>
    item_transform_perspective(sp, ep) |>
    item_transform_hflip()

  expect_s3_class(result, "image_with_bounding_box")
  # after shifting right by 10 then flipping horizontally: x -> 200 - (x + 10)
  expect_equal_to_r(result$y$boxes[1, 1], 200 - 60)
  expect_equal_to_r(result$y$boxes[1, 3], 200 - 20)
  expect_equal_to_r(result$y$boxes[1, 2], 20)
  expect_equal_to_r(result$y$boxes[1, 4], 60)
})

test_that("item_transform_perspective applies to datasets", {
  sp <- list(c(0, 0), c(199, 0), c(199, 99), c(0, 99))
  ep <- list(c(10, 20), c(209, 20), c(209, 119), c(10, 119))

  ds <- torch::dataset(
    name = "perspective_test",
    initialize = function() {
      self$items <- list(
        make_detection_item(matrix(c(10, 10, 20, 20), ncol = 4)),
        make_detection_item(matrix(c(5, 5, 15, 15), ncol = 4))
      )
    },
    .getitem = function(index) self$items[[index]],
    .length = function() length(self$items)
  )()

  transformed <- item_transform_perspective(ds, sp, ep)

  item1 <- transformed$.getitem(1)
  item2 <- transformed$.getitem(2)

  expect_s3_class(item1, "image_with_bounding_box")
  expect_equal_to_r(item1$y$boxes[1, 1], 20)
  expect_equal_to_r(item1$y$boxes[1, 2], 30)
  expect_equal_to_r(item1$y$boxes[1, 3], 30)
  expect_equal_to_r(item1$y$boxes[1, 4], 40)
  expect_equal_to_r(item2$y$boxes[1, 1], 15)
  expect_equal_to_r(item2$y$boxes[1, 2], 25)
  expect_equal_to_r(item2$y$boxes[1, 3], 25)
  expect_equal_to_r(item2$y$boxes[1, 4], 35)
})

