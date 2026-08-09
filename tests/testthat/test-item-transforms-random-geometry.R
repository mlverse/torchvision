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
