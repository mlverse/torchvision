test_that("item_transform_random_horizontal_flip rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_horizontal_flip(img),
    "requires a dataset item"
  )
})

test_that("item_transform_random_horizontal_flip rejects numeric input", {
  expect_error(
    item_transform_random_horizontal_flip(42),
    "requires a dataset item"
  )
})

test_that("item_transform_random_horizontal_flip with p=0 never flips", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()

  result <- item_transform_random_horizontal_flip(item, p = 0)

  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$boxes, original_boxes))
})

test_that("item_transform_random_horizontal_flip with p=1 always flips", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()

  result <- item_transform_random_horizontal_flip(item, p = 1)

  expect_true(torch_equal(result$x, transform_hflip(original_img)))
  expect_equal_to_r(result$y$boxes[1, 1], 200 - 50)
  expect_equal_to_r(result$y$boxes[1, 3], 200 - 10)
})

test_that("item_transform_random_horizontal_flip with p=0 does not mutate segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_img <- item$x$clone()
  original_masks <- item$y$masks$clone()

  result <- item_transform_random_horizontal_flip(item, p = 0)

  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$masks, original_masks))
})

test_that("item_transform_random_horizontal_flip with p=1 works for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_random_horizontal_flip(item, p = 1)

  expect_true(result$y$masks$equal(original_masks$flip(-1)))
})

test_that("item_transform_random_horizontal_flip with p=1 works for rotated boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_random_horizontal_flip(rotated, p = 1)

  expect_s3_class(result, "image_with_rotated_box")
  expect_equal_to_r(result$y$boxes[1, 5], -30, tolerance = 1e-5)
})

test_that("item_transform_random_horizontal_flip default p is 0.5", {
  fmls <- formals(item_transform_random_horizontal_flip)
  expect_equal(fmls$p, 0.5)
})

# --- item_transform_random_vertical_flip ---

test_that("item_transform_random_vertical_flip rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_vertical_flip(img),
    "requires a dataset item"
  )
})

test_that("item_transform_random_vertical_flip rejects numeric input", {
  expect_error(
    item_transform_random_vertical_flip(42),
    "requires a dataset item"
  )
})

test_that("item_transform_random_vertical_flip with p=0 never flips", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()

  result <- item_transform_random_vertical_flip(item, p = 0)

  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$boxes, original_boxes))
})

test_that("item_transform_random_vertical_flip with p=1 always flips", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()

  result <- item_transform_random_vertical_flip(item, p = 1)

  expect_true(torch_equal(result$x, transform_vflip(original_img)))
  expect_equal_to_r(result$y$boxes[1, 2], 100 - 60)
  expect_equal_to_r(result$y$boxes[1, 4], 100 - 20)
})

test_that("item_transform_random_vertical_flip with p=0 does not mutate segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_img <- item$x$clone()
  original_masks <- item$y$masks$clone()

  result <- item_transform_random_vertical_flip(item, p = 0)

  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$masks, original_masks))
})

test_that("item_transform_random_vertical_flip with p=1 works for segmentation", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_random_vertical_flip(item, p = 1)

  expect_true(result$y$masks$equal(original_masks$flip(-2)))
})

test_that("item_transform_random_vertical_flip with p=1 works for rotated boxes", {
  boxes <- matrix(c(10, 20, 50, 60), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_random_vertical_flip(rotated, p = 1)

  expect_s3_class(result, "image_with_rotated_box")
  expect_equal_to_r(result$y$boxes[1, 5], -30, tolerance = 1e-5)
})

test_that("item_transform_random_vertical_flip default p is 0.5", {
  fmls <- formals(item_transform_random_vertical_flip)
  expect_equal(fmls$p, 0.5)
})

# --- item_transform_random_crop ---

test_that("item_transform_random_crop rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_crop(img, size = c(50, 80)),
    "requires a dataset item"
  )
})

test_that("item_transform_random_crop rejects numeric input", {
  expect_error(
    item_transform_random_crop(42, size = c(50, 80)),
    "requires a dataset item"
  )
})

test_that("item_transform_random_crop default parameters", {
  fmls <- formals(item_transform_random_crop)
  expect_true(is.null(fmls$padding))
  expect_false(fmls$pad_if_needed)
  expect_equal(fmls$fill, 0)
  expect_equal(fmls$padding_mode, "constant")
})

test_that("item_transform_random_crop with size equal to image size returns the item unchanged", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_crop(item, size = c(100, 200))

  expect_true(torch_equal(result$x, item$x))
  expect_true(torch_equal(result$y$boxes, item$y$boxes))
})

test_that("item_transform_random_crop crops detection items and updates targets", {
  # box covering the whole image stays the full crop whatever the random offset
  item <- make_detection_item(matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_crop(item, size = c(50, 80))

  expect_tensor_shape(result$x, c(3, 50, 80))
  expect_equal(result$y$image_height, 50)
  expect_equal(result$y$image_width, 80)
  expect_equal_to_r(result$y$boxes[1, 1], 0)
  expect_equal_to_r(result$y$boxes[1, 2], 0)
  expect_equal_to_r(result$y$boxes[1, 3], 80)
  expect_equal_to_r(result$y$boxes[1, 4], 50)
})

test_that("item_transform_random_crop clips partially cropped boxes", {
  item <- make_detection_item(matrix(c(20, 30, 60, 70), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_crop(item, size = c(50, 80))

  boxes <- as.matrix(result$y$boxes$to(device = "cpu"))
  expect_true(all(boxes[, 1] >= 0 & boxes[, 1] < boxes[, 3] & boxes[, 3] <= 80))
  expect_true(all(boxes[, 2] >= 0 & boxes[, 2] < boxes[, 4] & boxes[, 4] <= 50))
})

test_that("item_transform_random_crop accepts a single int size for a square crop", {
  item <- make_detection_item(matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_crop(item, size = 80)

  expect_tensor_shape(result$x, c(3, 80, 80))
})

test_that("item_transform_random_crop works for segmentation items", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  item$y$masks$fill_(TRUE)

  result <- item_transform_random_crop(item, size = c(50, 80))

  expect_tensor_shape(result$x, c(3, 50, 80))
  expect_tensor_shape(result$y$masks, c(2, 50, 80))
  expect_equal_to_r(result$y$masks$min(), TRUE)
})

test_that("item_transform_random_crop works for rotated boxes", {
  item <- make_detection_item(matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))
  rotated <- item_transform_rotate(item, angle = 30)

  result <- item_transform_random_crop(rotated, size = c(50, 80))

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$x, c(3, 50, 80))
  expect_equal_to_r(result$y$boxes[1, 5], 30, tolerance = 1e-5)
  expect_equal_to_r(result$y$boxes[1, 3], 80)
  expect_equal_to_r(result$y$boxes[1, 4], 50)
})

test_that("item_transform_random_crop pads smaller images when pad_if_needed", {
  item <- make_detection_item(matrix(c(5, 5, 20, 25), ncol = 4), image_size = c(30L, 40L))

  result <- item_transform_random_crop(item, size = c(50, 60), pad_if_needed = TRUE)

  expect_tensor_shape(result$x, c(3, 50, 60))
})

test_that("item_transform_random_crop applies padding before cropping", {
  item <- make_detection_item(matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_crop(item, size = c(50, 80), padding = 10)

  expect_tensor_shape(result$x, c(3, 50, 80))
})

test_that("item_transform_random_crop errors when crop is larger than the image", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  expect_error(
    item_transform_random_crop(item, size = c(300, 400)),
    "Required crop size"
  )
})

test_that("item_transform_random_crop works on a detection dataset", {
  ds <- dataset(
    name = "toy_detection",
    initialize = function() {},
    .getitem = function(index) {
      make_detection_item(matrix(c(0, 0, 200, 100), ncol = 4), image_size = c(100L, 200L))
    },
    .length = function() 1L
  )()

  ds <- item_transform_random_crop(ds, size = c(50, 80))
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_bounding_box")
  expect_tensor_shape(item$x, c(3, 50, 80))
  expect_equal_to_r(item$y$boxes[1, 3], 80)
  expect_equal_to_r(item$y$boxes[1, 4], 50)
})

test_that("item_transform_random_crop works on a segmentation dataset", {
  ds <- dataset(
    name = "toy_segmentation",
    initialize = function() {},
    .getitem = function(index) {
      item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
      item$y$masks$fill_(TRUE)
      item
    },
    .length = function() 1L
  )()

  ds <- item_transform_random_crop(ds, size = c(50, 80))
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_segmentation_mask")
  expect_tensor_shape(item$x, c(3, 50, 80))
  expect_tensor_shape(item$y$masks, c(2, 50, 80))
})
