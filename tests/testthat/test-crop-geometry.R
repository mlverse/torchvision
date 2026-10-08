make_crop_marker_item <- function(image_size = c(16L, 20L),
                                  box = c(8, 6, 12, 10)) {
  item <- make_detection_item(matrix(box, nrow = 1), image_size = image_size)
  item$x$zero_()
  item$x[2, (box[2] + 1):box[4], (box[1] + 1):box[3]] <- 1
  item
}

expect_crop_marker_alignment <- function(item) {
  pixels <- which(as_array(item$x[2, , ]) == 1, arr.ind = TRUE)
  expect_equal_to_r(item$y$boxes[1, 1:4], c(
    min(pixels[, 2]) - 1, min(pixels[, 1]) - 1,
    max(pixels[, 2]), max(pixels[, 1])
  ))
}

test_that("padded center crops shift boxes with the image content", {
  item <- make_crop_marker_item(c(100L, 100L), c(10, 10, 20, 20))
  result <- item_transform_center_crop(item, 120)

  expect_tensor_shape(result$x, c(3, 120, 120))
  expect_equal_to_r(result$y$boxes[1, ], c(20, 20, 30, 30))
  expect_crop_marker_alignment(result)
  expect_equal_to_r(result$x[2, , ]$sum(), 100)
  expect_equal_to_r(item$y$boxes[1, ], c(10, 10, 20, 20))
  expect_crop_marker_alignment(item)
})

test_that("center crops align pixels and boxes with rectangular and odd padding", {
  sizes <- list(c(24, 30), c(25, 31), c(24, 20), c(16, 30),
                c(24, 10), c(8, 30), c(16, 20))
  shifts <- list(c(5, 4), c(5, 4), c(0, 4), c(5, 0),
                 c(-4, 4), c(5, -3), c(0, 0))

  for (rotated in c(FALSE, TRUE)) {
    item <- make_crop_marker_item()
    if (rotated) {
      item <- item_transform_rotate(item, 90, interpolation = 0, expand = FALSE)
    }
    original_boxes <- as_array(item$y$boxes)
    original_pixels <- as_array(item$x)

    for (i in seq_along(sizes)) {
      result <- item_transform_center_crop(item, sizes[[i]])
      expected <- original_boxes[1, 1:4] + rep(shifts[[i]], 2)
      expect_equal_to_r(result$y$boxes[1, 1:4], expected)
      expect_crop_marker_alignment(result)
      expect_equal_to_r(result$x[2, , ]$sum(), 16)
      expect_tensor_shape(result$x, c(3, sizes[[i]]))
      expect_equal(c(result$y$image_height, result$y$image_width), sizes[[i]])
      expect_equal_to_r(result$y$labels, as_array(item$y$labels))
      if (rotated) {
        expect_s3_class(result, "image_with_rotated_box")
        expect_equal_to_r(result$y$boxes[1, 5], 90)
      }
    }
    expect_equal_to_r(item$x, original_pixels)
    expect_equal_to_r(item$y$boxes, original_boxes)
  }
})

test_that("padded center crops preserve alignment in transform compositions", {
  item <- make_crop_marker_item(box = c(4, 5, 8, 9))
  result <- item |>
    item_transform_hflip() |>
    item_transform_center_crop(c(24, 30)) |>
    item_transform_vflip() |>
    item_transform_crop(top = 3, left = 4, height = 20, width = 24)

  expect_equal_to_r(result$y$boxes[1, ], c(14, 9, 18, 13))
  expect_crop_marker_alignment(result)
  expect_tensor_shape(result$x, c(3, 20, 24))
})

test_that("padded center crops keep segmentation pixels and masks aligned", {
  item <- make_segmentation_item(c(16L, 20L), num_masks = 1L)
  item$x$zero_()
  item$x[1, 7:10, 9:12] <- 1
  item$y$masks <- item$x[1, , , drop = FALSE]$to(dtype = torch_bool())

  result <- item_transform_center_crop(item, c(25, 31))

  expect_equal_to_r(result$y$masks[1, , ], as_array(result$x[1, , ]) == 1)
  expect_equal_to_r(result$y$masks$sum(), 16)
})

test_that("random crops select every valid integer origin including the last", {
  withr::local_seed(42)
  x <- torch_tensor(array(seq_len(20), c(1, 4, 5)))
  sizes <- list(c(3, 4), c(4, 4), c(3, 5), c(4, 5))
  expected_origins <- list(c(1, 2, 5, 6), c(1, 5), c(1, 2), 1)

  for (i in seq_along(sizes)) {
    origins <- replicate(32, {
      crop <- transform_random_crop(x, sizes[[i]])
      expect_tensor_shape(crop, c(1, sizes[[i]]))
      origin <- as.numeric(crop[1, 1, 1])
      top <- (origin - 1) %% 4 + 1
      left <- (origin - 1) %/% 4 + 1
      expect_equal_to_r(crop, as_array(x[
        , top:(top + sizes[[i]][1] - 1), left:(left + sizes[[i]][2] - 1),
        drop = FALSE
      ]))
      origin
    })
    expect_equal(sort(unique(origins)), expected_origins[[i]])
  }
})

test_that("random crops translate boxes by the exact sampled pixel origin", {
  withr::local_seed(123)
  for (rotated in c(FALSE, TRUE)) {
    item <- make_crop_marker_item()
    if (rotated) {
      item <- item_transform_rotate(item, 90, interpolation = 0, expand = FALSE)
    }
    # The first channel records the one-based row and column of each pixel.
    item$x[1, , ] <- torch_tensor(outer(1:16, 1:20, function(row, col) 100 * row + col))
    for (i in seq_len(16)) {
      result <- item_transform_random_crop(item, c(12, 16))
      origin <- as.numeric(result$x[1, 1, 1])
      top <- origin %/% 100
      left <- origin %% 100
      expect_equal_to_r(result$y$boxes[1, 1:4],
                        c(8, 6, 12, 10) - rep(c(left - 1, top - 1), 2))
      expect_crop_marker_alignment(result)
      if (rotated) expect_equal_to_r(result$y$boxes[1, 5], 90)
    }
  }
})

test_that("five crops use the requested size and the four corner contents", {
  for (batched in c(FALSE, TRUE)) {
    x <- torch_tensor(array(seq_len(2 * 3 * 6 * 10), c(2, 3, 6, 10)))
    if (!batched) x <- x[1, ]
    for (transposed in c(FALSE, TRUE)) {
      if (transposed) x <- x$transpose(-2, -1)
      size <- if (transposed) c(4, 2) else c(2, 4)
      expected <- if (transposed) {
        list(x[.., 1:4, 1:2], x[.., 1:4, 5:6],
             x[.., 7:10, 1:2], x[.., 7:10, 5:6], x[.., 3:6, 2:3])
      } else {
        list(x[.., 1:2, 1:4], x[.., 1:2, 7:10],
             x[.., 5:6, 1:4], x[.., 5:6, 7:10], x[.., 2:3, 3:6])
      }
      crops <- transform_five_crop(x, size)
      expect_length(crops, 5)
      for (i in seq_along(crops)) {
        expect_tensor_shape(crops[[i]], c(if (batched) c(2, 3) else 3, size))
        expect_equal_to_r(crops[[i]], as_array(expected[[i]]))
      }
    }
  }
})

test_that("ten crops return five requested crops from each flip orientation", {
  for (batched in c(FALSE, TRUE)) {
    x <- torch_tensor(array(seq_len(2 * 3 * 6 * 10), c(2, 3, 6, 10)))
    if (!batched) x <- x[1, ]
    for (vertical in c(FALSE, TRUE)) {
      flipped <- if (vertical) x$flip(-2) else x$flip(-1)
      expected <- c(transform_five_crop(x, c(2, 4)),
                    transform_five_crop(flipped, c(2, 4)))
      crops <- transform_ten_crop(x, c(2, 4), vertical_flip = vertical)
      expect_length(crops, 10)
      for (i in seq_along(crops)) {
        expect_tensor_shape(crops[[i]], c(if (batched) c(2, 3) else 3, 2, 4))
        expect_equal_to_r(crops[[i]], as_array(expected[[i]]))
      }
    }
  }
})
