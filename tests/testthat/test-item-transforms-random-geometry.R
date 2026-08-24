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

# item_transform_random_vertical_flip

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


# item_transform_random_crop

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
  expect_equal_to_r(result$y$boxes[1, 5], 30)
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


# item_transform_random_affine

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


# item_transform_random_erasing

test_that("item_transform_random_erasing rejects non-item inputs", {
  img <- torch_randn(3, 100, 200)
  expect_error(
    item_transform_random_erasing(img),
    "requires a dataset item"
  )
  expect_error(
    item_transform_random_erasing(42),
    "requires a dataset item"
  )
})

test_that("item_transform_random_erasing works on detection items", {
  item <- make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))

  # p = 0 never erases
  original_img <- item$x$clone()
  original_boxes <- item$y$boxes$clone()
  result <- item_transform_random_erasing(item, p = 0)
  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$boxes, original_boxes))

  # p = 1 erases the image but keeps boxes unchanged
  result <- item_transform_random_erasing(item, p = 1)
  expect_true(torch_equal(result$y$boxes, original_boxes))
  expect_false(torch_equal(result$x, original_img))
  expect_true((result$x == 0)$any()$item())

  # the input item is never mutated
  expect_true(torch_equal(item$x, original_img))
  expect_s3_class(result, "image_with_bounding_box")

  # per-channel value
  result <- item_transform_random_erasing(item, p = 1, value = c(1, 2, 3))
  region <- (result$x[1, , ] == 1) & (result$x[2, , ] == 2) & (result$x[3, , ] == 3)
  expect_true(region$any()$item())

  # random value
  result <- item_transform_random_erasing(item, p = 1, value = "random")
  expect_false(torch_equal(result$x, item$x))

  # rotated-box items
  rotated <- item_transform_rotate(item, angle = 30)
  result <- item_transform_random_erasing(rotated, p = 1, value = c(5, 0.1, 7e9), ratio = c(0.1, 3.9), scale = c(0.1, 3.9))
  expect_s3_class(result, "image_with_rotated_box")
  expect_true(torch_equal(result$y$boxes, rotated$y$boxes))
})

test_that("item_transform_random_erasing works on segmentation items", {
  item <- make_segmentation_item(image_size = c(100L, 200L), num_masks = 2L)
  original_img <- item$x$clone()
  original_masks <- item$y$masks$clone()

  # p = 0 never erases
  result <- item_transform_random_erasing(item, p = 0)
  expect_true(torch_equal(result$x, original_img))
  expect_true(torch_equal(result$y$masks, original_masks))

  # p = 1 erases the image but keeps masks unchanged
  result <- item_transform_random_erasing(item, p = 1)
  expect_true(torch_equal(result$y$masks, original_masks))
  expect_false(torch_equal(result$x, item$x))
})

test_that("item_transform_random_erasing works on a dataset", {
  ds <- dataset(
    name = "toy_detection",
    initialize = function() {},
    .getitem = function(index) {
      make_detection_item(matrix(c(10, 20, 50, 60), ncol = 4), image_size = c(100L, 200L))
    },
    .length = function() 1L
  )()

  transformed <- item_transform_random_erasing(ds, p = 0)
  item <- transformed$.getitem(1)

  expect_s3_class(item, "image_with_bounding_box")
})

test_that("item_transform_random_erasing default parameters", {
  fmls <- formals(item_transform_random_erasing)
  expect_equal(fmls$p, 0.5)
  expect_equal(eval(fmls$scale), c(0.02, 0.33))
  expect_equal(eval(fmls$ratio), c(0.3, 3.3))
  expect_equal(fmls$value, 0)
  expect_false(fmls$inplace)
})

