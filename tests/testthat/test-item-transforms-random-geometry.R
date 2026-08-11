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

test_that("item_transform_random_resize_crop rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_resize_crop(img, size = c(50, 100)),
    "requires a dataset item"
  )
  expect_error(
    item_transform_random_resize_crop(42, size = c(50, 100)),
    "requires a dataset item"
  )
})

test_that("item_transform_random_resize_crop returns detection items at the requested size", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))
  original_img <- item$x$clone()
  original_boxes <- as_array(item$y$boxes)

  set.seed(1)
  torch_manual_seed(1)
  result <- item_transform_random_resize_crop(item, size = c(50L, 60L))

  expect_s3_class(result, "image_with_bounding_box")
  expect_tensor_dtype(result$x, item$x$dtype)
  expect_tensor_shape(result$x, c(3, 50, 60))
  expect_equal(result$y$image_height, 50L)
  expect_equal(result$y$image_width, 60L)
  expect_tensor_shape(result$y$boxes, c(1, 4))
  expect_true((result$y$boxes >= 0)$all()$item())
  expect_true((result$y$boxes[, 3] <= 60)$all()$item())
  expect_true((result$y$boxes[, 4] <= 50)$all()$item())

  expect_true(torch_equal(item$x, original_img))
  expect_equal_to_r(item$y$boxes, original_boxes)
})

test_that("item_transform_random_resize_crop rescales boxes to the output size", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 100L))

  result <- item_transform_random_resize_crop(item, size = c(50L, 200L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_tensor_shape(result$x, c(3, 50, 200))
  expect_true(torch_equal(result$x, transform_resize(item$x, size = c(50L, 200L))))
  expect_equal_to_r(result$y$boxes[1, ], c(40, 15, 160, 45))
  expect_equal(result$y$image_height, 50L)
  expect_equal(result$y$image_width, 200L)
})

test_that("item_transform_random_resize_crop matches the smaller edge for a bare integer size", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))

  result <- item_transform_random_resize_crop(item, size = 50, scale = c(1, 1), ratio = c(2, 2))

  expect_tensor_shape(result$x, c(3, 50, 100))
  expect_equal_to_r(result$y$boxes[1, ], c(10, 15, 40, 45))
  expect_equal(result$y$image_height, 50L)
  expect_equal(result$y$image_width, 100L)
})

test_that("item_transform_random_resize_crop matches the crop and the resize it composes", {
  boxes <- matrix(c(20, 30, 80, 90, 120, 40, 190, 95), ncol = 4, byrow = TRUE)
  item <- make_detection_item(boxes, image_size = c(100L, 200L))

  set.seed(42)
  torch_manual_seed(42)
  params <- get_random_resized_crop_params(item$x, c(0.08, 1.0), c(3 / 4, 4 / 3))

  set.seed(42)
  torch_manual_seed(42)
  result <- item_transform_random_resize_crop(item, size = c(50L, 60L))

  cropped <- item_transform_crop(item, params[1], params[2], params[3], params[4])
  expected_img <- transform_resize(cropped$x, size = c(50L, 60L))
  expected_boxes <- sweep(
    as_array(cropped$y$boxes), 2,
    c(60 / params[4], 50 / params[3], 60 / params[4], 50 / params[3]), "*"
  )

  expect_true(torch_equal(result$x, expected_img))
  expect_equal_to_r(result$y$boxes, expected_boxes, tolerance = 1e-5)
})

test_that("item_transform_random_resize_crop crops the image edges on the central-crop fallback", {
  item <- make_detection_item(matrix(c(60, 10, 90, 40), ncol = 4), image_size = c(100L, 200L))

  expect_equal(get_random_resized_crop_params(item$x, c(1, 1), c(1, 1)), c(1, 51, 100, 100))

  result <- item_transform_random_resize_crop(item, size = c(100L, 100L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_true(torch_equal(result$x, transform_resize(item$x[, , 51:150], size = c(100L, 100L))))
  expect_equal_to_r(result$y$boxes[1, ], c(10, 10, 40, 40))
})

test_that("item_transform_random_resize_crop preserves labels and handles empty boxes", {
  labels <- torch_tensor(c(1L, 2L), dtype = torch_long())
  item <- make_detection_item(
    matrix(c(20, 30, 80, 90, 30, 40, 90, 95), ncol = 4, byrow = TRUE),
    labels = labels,
    image_size = c(100L, 100L)
  )
  result <- item_transform_random_resize_crop(item, size = c(50L, 50L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_equal_to_r(result$y$labels, c(1L, 2L))

  item <- make_detection_item(
    boxes = matrix(numeric(0), ncol = 4),
    labels = torch_zeros(0L, dtype = torch_long()),
    image_size = c(100L, 100L)
  )
  result <- item_transform_random_resize_crop(item, size = c(50L, 50L))

  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
})

test_that("item_transform_random_resize_crop rescales the target area", {
  item <- make_detection_item(matrix(c(20, 30, 80, 90), ncol = 4), image_size = c(100L, 100L))

  result <- item_transform_random_resize_crop(item, size = c(50L, 50L),
                                              scale = c(1, 1), ratio = c(1, 1))

  box <- as_array(result$y$boxes)[1, ]
  expect_equal(as.numeric(result$y$area), (box[3] - box[1]) * (box[4] - box[2]))
  expect_equal(as.numeric(result$y$area), as.numeric(item$y$area) / 4)

  result <- item_transform_random_resize_crop(item, size = c(50L, 200L),
                                              scale = c(1, 1), ratio = c(1, 1))
  expect_equal(as.numeric(result$y$area), as.numeric(item$y$area))
})

test_that("item_transform_random_resize_crop drops boxes falling outside the crop", {
  item <- make_detection_item(matrix(c(150, 10, 190, 40), ncol = 4), image_size = c(100L, 200L))

  result <- item_transform_random_resize_crop(item, size = c(30L, 30L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_tensor_shape(result$x, c(3, 30, 30))
  expect_tensor_shape(result$y$boxes, c(0, 4))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_tensor_shape(result$y$labels, 0)
})

test_that("item_transform_random_resize_crop crops and resizes segmentation items", {
  item <- make_segmentation_item(image_size = c(100L, 100L), num_masks = 2L)
  original_masks <- item$y$masks$clone()
  original_labels <- as_array(item$y$labels)

  result <- item_transform_random_resize_crop(item, size = c(50L, 200L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_s3_class(result, "image_with_segmentation_mask")
  expect_tensor_shape(result$x, c(3, 50, 200))
  expect_tensor_shape(result$y$masks, c(2, 50, 200))
  expect_tensor_dtype(result$y$masks, torch_bool())
  expect_equal_to_r(result$y$labels, original_labels)
  expect_equal(result$y$image_height, 50L)
  expect_equal(result$y$image_width, 200L)

  expected_masks <- transform_resize(original_masks, size = c(50L, 200L), interpolation = 0)
  expect_true(result$y$masks$equal(expected_masks))
})

test_that("item_transform_random_resize_crop matches the crop and the resize it composes on masks", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)

  set.seed(5)
  torch_manual_seed(5)
  params <- get_random_resized_crop_params(item$x, c(0.08, 1.0), c(3 / 4, 4 / 3))

  set.seed(5)
  torch_manual_seed(5)
  result <- item_transform_random_resize_crop(item, size = c(64L, 64L))

  cropped <- item_transform_crop(item, params[1], params[2], params[3], params[4])

  expect_true(torch_equal(result$x, transform_resize(cropped$x, size = c(64L, 64L))))
  expect_true(torch_equal(
    result$y$masks,
    transform_resize(cropped$y$masks, size = c(64L, 64L), interpolation = 0)
  ))
})

test_that("item_transform_random_resize_crop handles items without any mask", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 0L)

  set.seed(3)
  torch_manual_seed(3)
  result <- item_transform_random_resize_crop(item, size = c(50L, 60L))

  expect_tensor_shape(result$x, c(3, 50, 60))
  expect_tensor_shape(result$y$masks, c(0, 50, 60))
  expect_tensor_dtype(result$y$masks, torch_bool())
})

test_that("item_transform_random_resize_crop resizes a two-dimensional mask", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 1L)
  item$y$masks <- item$y$masks$squeeze(1)

  set.seed(9)
  torch_manual_seed(9)
  result <- item_transform_random_resize_crop(item, size = c(50L, 60L))

  expect_tensor_shape(result$y$masks, c(50, 60))
  expect_tensor_dtype(result$y$masks, torch_bool())
})

test_that("item_transform_random_resize_crop handles rotated boxes", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 100L))
  rotated <- item_transform_rotate(item, angle = 30, expand = FALSE)

  result <- item_transform_random_resize_crop(rotated, size = c(50L, 50L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expect_s3_class(result, "image_with_rotated_box")
  expect_tensor_shape(result$y$boxes, c(1, 5))
  expect_tensor_dtype(result$y$boxes, torch_float())
  expect_equal_to_r(result$y$boxes[1, 5], 30, tolerance = 1e-5)
  clipped <- pmin(as_array(rotated$y$boxes)[1, 1:4], 100)
  expect_equal_to_r(result$y$boxes[1, 1:4], clipped / 2, tolerance = 1e-5)
})

test_that("item_transform_random_resize_crop tilts rotated boxes when the edges scale unequally", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 100L))
  rotated <- item_transform_rotate(item, angle = 30, expand = FALSE)

  result <- item_transform_random_resize_crop(rotated, size = c(50L, 200L),
                                              scale = c(1, 1), ratio = c(1, 1))

  expected_angle <- rad2deg(atan2(0.5 * sin(deg2rad(30)), 2 * cos(deg2rad(30))))
  expect_equal_to_r(result$y$boxes[1, 5], expected_angle, tolerance = 1e-5)
  clipped <- pmin(as_array(rotated$y$boxes)[1, 1:4], 100)
  expect_equal_to_r(result$y$boxes[1, 1:4], clipped * c(2, 0.5, 2, 0.5), tolerance = 1e-5)
})

test_that("item_transform_random_resize_crop can be composed", {
  boxes <- matrix(c(20, 30, 80, 90), ncol = 4)
  item <- make_detection_item(boxes, image_size = c(100L, 100L))

  result <- item |>
    item_transform_random_resize_crop(size = c(50L, 200L), scale = c(1, 1), ratio = c(1, 1)) |>
    item_transform_hflip()

  expect_s3_class(result, "image_with_bounding_box")
  expect_tensor_shape(result$x, c(3, 50, 200))
  expect_equal_to_r(result$y$boxes[1, ], c(200 - 160, 15, 200 - 40, 45))
})

test_that("item_transform_random_resize_crop works on detection and segmentation datasets", {
  detection_item <- make_detection_item(matrix(c(20, 30, 80, 90), ncol = 4), image_size = c(100L, 200L))
  ds <- dataset(
    name = "toy_detection",
    initialize = function() {},
    .getitem = function(index) detection_item,
    .length = function() 1L
  )()

  ds <- item_transform_random_resize_crop(ds, size = c(50L, 60L))

  set.seed(7)
  torch_manual_seed(7)
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_bounding_box")
  expect_tensor_shape(item$x, c(3, 50, 60))
  expect_equal(item$y$image_height, 50L)
  expect_equal(item$y$image_width, 60L)

  other <- ds$.getitem(1)
  expect_false(torch_equal(item$x, other$x))
  expect_equal_to_r(detection_item$y$boxes, matrix(c(20, 30, 80, 90), ncol = 4))

  ds <- dataset(
    name = "toy_segmentation",
    initialize = function() {},
    .getitem = function(index) {
      make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
    },
    .length = function() 1L
  )()

  ds <- item_transform_random_resize_crop(ds, size = c(50L, 60L))
  item <- ds$.getitem(1)

  expect_s3_class(item, "image_with_segmentation_mask")
  expect_tensor_shape(item$x, c(3, 50, 60))
  expect_tensor_shape(item$y$masks, c(2, 50, 60))
})
