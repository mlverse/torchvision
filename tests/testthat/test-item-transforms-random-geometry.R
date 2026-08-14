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

# --- item_transform_random_affine ---

test_that("item_transform_random_affine rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_affine(img, degrees = 30),
    "requires a dataset item"
  )
  expect_error(
    item_transform_random_affine(42, degrees = 30),
    "requires a dataset item"
  )
})

test_that("item_transform_random_affine validates its ranges", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))

  expect_error(item_transform_random_affine(item, degrees = -10), "degrees must be positive")
  expect_error(item_transform_random_affine(item, degrees = c(-10, 0, 10)), "degrees must be length 1 or 2")
  expect_error(item_transform_random_affine(item, degrees = 10, translate = 0.1), "translate must be length 2")
  expect_error(item_transform_random_affine(item, degrees = 10, translate = c(0.1, 2)), "translate must be between 0 and 1")
  expect_error(item_transform_random_affine(item, degrees = 10, scale = 0.5), "scale must be length 2")
  expect_error(item_transform_random_affine(item, degrees = 10, scale = c(-1, 1)), "scale must be positive")
  expect_error(item_transform_random_affine(item, degrees = 10, shear = -5), "shear must be positive")
  expect_error(item_transform_random_affine(item, degrees = 10, shear = c(1, 2, 3)), "shear's length must be 1, 2, or 4")
})

test_that("item_transform_random_affine with a zero range is the identity", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  original_img <- item$x$clone()

  result <- item_transform_random_affine(item, degrees = 0)

  expect_s3_class(result, "image_with_rotated_box")
  expect_true(torch_equal(result$x, original_img))
  expect_equal_to_r(result$y$boxes, matrix(c(10, 20, 50, 60, 0), ncol = 5))
})

test_that("item_transform_random_affine matches the affine it composes", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  set.seed(1)
  torch_manual_seed(1)
  params <- get_random_affine_params(c(-30, 30), c(0.1, 0.2), c(0.8, 1.2), c(-10, 10),
                                     get_image_size(item$x))

  set.seed(1)
  torch_manual_seed(1)
  result <- item_transform_random_affine(item, degrees = 30, translate = c(0.1, 0.2),
                                         scale = c(0.8, 1.2), shear = 10)

  expected <- item_transform_affine(item, angle = params[[1]], translate = params[[2]],
                                    scale = params[[3]], shear = params[[4]])

  expect_true(torch_equal(result$x, expected$x))
  expect_equal_to_r(result$y$boxes, as_array(expected$y$boxes))
})

test_that("item_transform_random_affine draws the angle inside the given range", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  angles <- vapply(1:20, function(i) {
    as.numeric(item_transform_random_affine(item, degrees = 30)$y$boxes[1, 5])
  }, numeric(1))

  expect_true(all(angles >= -30 & angles <= 30))
  expect_gt(length(unique(angles)), 1)
})

test_that("item_transform_random_affine draws the translation inside the given range", {
  item <- make_detection_item(matrix(c(80, 40, 120, 60), ncol = 4), image_size = c(100L, 200L))

  shifts <- vapply(1:20, function(i) {
    boxes <- item_transform_random_affine(item, degrees = 0, translate = c(0.1, 0.2))$y$boxes
    c(as.numeric(boxes[1, 1]) - 80, as.numeric(boxes[1, 2]) - 40)
  }, numeric(2))

  expect_true(all(abs(shifts[1, ]) <= 0.1 * 200))
  expect_true(all(abs(shifts[2, ]) <= 0.2 * 100))
})

test_that("item_transform_random_affine keeps the image size and dtype for detection", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_affine(item, degrees = 30, translate = c(0.1, 0.1),
                                         scale = c(0.8, 1.2), shear = 10)

  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_tensor_dtype(result$x, item$x$dtype)
  expect_tensor_shape(result$y$boxes, c(1, 5))
})

test_that("item_transform_random_affine preserves labels and handles empty boxes", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(10, 20, 50, 60, 5, 5, 15, 25), ncol = 4, byrow = TRUE),
    labels = labels
  )
  result <- item_transform_random_affine(item, degrees = 30)

  expect_true(result$y$labels$eq(labels)$all()$item())

  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long())
  )
  result <- item_transform_random_affine(item, degrees = 30)

  expect_tensor_shape(result$y$boxes, c(0, 5))
})

test_that("item_transform_random_affine does not mutate its input", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4))
  original_img <- item$x$clone()
  original_boxes <- as_array(item$y$boxes)

  item_transform_random_affine(item, degrees = 30, translate = c(0.1, 0.1))

  expect_true(torch_equal(item$x, original_img))
  expect_equal_to_r(item$y$boxes, original_boxes)
})

test_that("item_transform_random_affine transforms segmentation masks", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_masks <- item$y$masks$clone()

  result <- item_transform_random_affine(item, degrees = c(45, 45))

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_tensor_shape(result$x, c(3, 100, 200))
  expect_tensor_shape(result$y$masks, original_masks$shape)
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_false(result$y$masks$equal(original_masks))
})

test_that("item_transform_random_affine keeps rotated boxes rotated", {
  item <- make_detection_item(matrix(c(20, 30, 80, 90), ncol = 4), image_size = c(100L, 100L))
  rotated <- item_transform_rotate(item, angle = 30, expand = FALSE)

  result <- item_transform_random_affine(rotated, degrees = c(0, 0))

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
})

test_that("item_transform_random_affine works on detection and segmentation datasets", {
  detection_item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
  ds <- dataset(
    name = "toy_detection",
    initialize = function() {},
    .getitem = function(index) detection_item,
    .length = function() 1L
  )()

  ds <- item_transform_random_affine(ds, degrees = 30, translate = c(0.1, 0.1))
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_rotated_box")
  expect_tensor_shape(item$x, c(3, 100, 200))

  other <- ds$.getitem(1)
  expect_false(torch_equal(item$x, other$x))
  expect_equal_to_r(detection_item$y$boxes, matrix(c(10, 20, 50, 60), ncol = 4))

  ds <- dataset(
    name = "toy_segmentation",
    initialize = function() {},
    .getitem = function(index) {
      make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
    },
    .length = function() 1L
  )()

  ds <- item_transform_random_affine(ds, degrees = 30)
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_segmentation_mask")
  expect_tensor_shape(item$y$masks, c(2, 100, 200))
})
