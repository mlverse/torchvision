x1 <- torch::torch_ones(5)
y1 <- torch::torch_ones(5)
x2 <- x1 + 1
y2 <- y1 + 1
boxes <- torch::torch_stack(list(x1,y1,x2,y2))$transpose(2,1)

test_that("batched_nms", {
  expect_no_error(
    x <- batched_nms(
      boxes = boxes,
      scores = torch::torch_ones(5)*0.6,
      idxs = torch::torch_ones(5),
      iou_threshold = 0.5
    )
  )
  expect_tensor(x)
  expect_equal_to_r(x, 1L)

  boxes <- torch::torch_stack(list(x1 + torch_randint(0,6,5),
                                  y1 + torch_randint(0,6,5),
                                  x2 + torch_randint(6,12,5),
                                  y2 + torch_randint(6,12,5)))$transpose(2,1)
  expect_no_error(
    x <- batched_nms(
      boxes = boxes,
      scores = torch::torch_ones(5)*0.6,
      idxs = torch::torch_ones(5),
      iou_threshold = 2
    )
  )
  expect_tensor(x)
  expect_gt(x$shape, 1L)
})

test_that("remove_small_boxes", {
  expect_no_error(x <- remove_small_boxes(boxes, 1))
  expect_tensor(x)
})

test_that("clip_boxes_to_image", {
  expect_no_error(x <- clip_boxes_to_image(boxes, c(10,10)))
  expect_tensor(x)
})

test_that("box_convert", {
  xyxy <- boxes
  xywh <- box_convert(xyxy, "xyxy", "xywh")
  cxcywh <- box_convert(xyxy, "xyxy", "cxcywh")

  expect_tensor(xywh)
  expect_tensor(cxcywh)
})

test_that("box_area", {
  area_6 <- torch::torch_tensor(matrix(c(0,0,2,3), 1, 4))
  expect_no_error(x <- box_area(area_6))
  expect_equal_to_r(x, 6)
})

test_that("box_iou", {
  expect_no_error(x <- box_iou(boxes, boxes))
  expect_tensor(x)
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))

  expect_no_error(x <- generalized_box_iou(boxes, boxes))
  expect_tensor(x)
  expect_equal_to_r(x, matrix(1, nrow = 5, ncol = 5))
})

test_that("nms basic functionality", {
  # Test with known overlapping boxes
  test_boxes <- torch::torch_tensor(rbind(
    c(10, 10, 50, 50),
    c(15, 15, 55, 55),  # High IoU with first box
    c(100, 100, 150, 150),
    c(105, 105, 155, 155)  # High IoU with third box
  ))
  test_scores <- torch::torch_tensor(c(0.9, 0.8, 0.7, 0.6))

  keep <- nms(test_boxes, test_scores, iou_threshold = 0.5)
  expect_tensor(keep)
  expect_equal_to_r(keep, c(1L, 3L))  # Keep highest scorers with low IoU

  # Test empty input
  empty_boxes <- torch::torch_empty(c(0, 4))
  empty_scores <- torch::torch_empty(0)
  keep_empty <- nms(empty_boxes, empty_scores, 0.5)
  expect_tensor(keep_empty)
  expect_equal(keep_empty$shape[1], 0)

  # Test single box
  single_box <- torch::torch_tensor(rbind(c(10, 10, 50, 50)))
  single_score <- torch::torch_tensor(c(0.9))
  keep_single <- nms(single_box, single_score, 0.5)
  expect_tensor(keep_single)
  expect_equal_to_r(keep_single, 1L)
})

test_that("nms R fallback matches optimized implementation", {
  # Create test data with various overlaps
  test_boxes <- torch::torch_tensor(rbind(
    c(10, 10, 50, 50),
    c(15, 15, 55, 55),   # Overlaps with box 1
    c(100, 100, 150, 150),
    c(20, 20, 60, 60),   # Overlaps with box 1
    c(110, 110, 160, 160) # Overlaps with box 3
  ))
  test_scores <- torch::torch_tensor(c(0.9, 0.8, 0.7, 0.75, 0.65))
  iou_threshold <- 0.5

  # Get result from current nms() which may use torchvisionlib if available
  result_optimized <- nms(test_boxes, test_scores, iou_threshold)

  # Manually run the R fallback implementation to verify it's correct
  n_boxes <- test_boxes$shape[1]
  sort_result <- test_scores$sort(descending = TRUE)
  order <- sort_result[[2]]
  sorted_boxes <- test_boxes[order, ]

  keep <- vector("integer", n_boxes)
  keep[1] <- 1L
  n_keep <- 1L

  for (i in 2:n_boxes) {
    current_box <- sorted_boxes[i, ]$unsqueeze(1)
    kept_indices <- keep[1:n_keep]
    kept_boxes <- sorted_boxes[kept_indices, , drop = FALSE]
    iou <- box_iou(kept_boxes, current_box)

    if (all(as.logical(iou <= iou_threshold))) {
      n_keep <- n_keep + 1L
      keep[n_keep] <- i
    }
  }

  kept_indices <- keep[1:n_keep]
  result_r_fallback <- order[kept_indices]

  # Both implementations should produce identical results
  expect_equal_to_r(result_optimized, as.array(result_r_fallback))
})

