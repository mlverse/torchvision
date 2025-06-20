x1 = torch::torch_ones(5)
y1 = torch::torch_ones(5)
x2 = x1 + 1
y2 = y1 + 1
boxes = torch::torch_stack(list(x1,y1,x2,y2))$transpose(2,1)

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

  boxes = torch::torch_stack(list(x1 + torch_randint(0,6,5),
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
  expect_true(x$shape > 1L)
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

