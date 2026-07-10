x1 <- torch::torch_ones(5)
y1 <- torch::torch_ones(5)
x2 <- x1 + 1
y2 <- y1 + 1
xyxy <- torch::torch_stack(list(x1,y1,x2,y2))$transpose(2,1)

test_that("box_cxcywh_to_xyxy box_xyxy_to_cxcywh box_xywh_to_xyxy box_xyxy_to_xywh", {
  # from xyxy
  cxcywh <- box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(xyxy))
  xywh <- box_xywh_to_xyxy(box_xyxy_to_xywh(xyxy))

  expect_tensor(cxcywh)
  expect_tensor(xywh)
  expect_equal(torch::as_array(cxcywh), torch::as_array(xyxy))
  expect_equal(torch::as_array(xywh), torch::as_array(xyxy))
})

test_that("box_xyxy_to_xyxyr adds zero rotation column", {
  result <- box_xyxy_to_xyxyr(xyxy)

  expect_tensor(result)
  expect_equal(result$size(2), 5L)
  expect_equal(result$size(1), 5L)
  expect_equal(torch::as_array(result[, 1:4]), torch::as_array(xyxy))
  expect_equal(torch::as_array(result[, 5]), rep(0, 5))
})

test_that("box_xyxy_to_xyxyr handles single box", {
  single <- torch::torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- box_xyxy_to_xyxyr(single)

  expect_equal(result$size(1), 1L)
  expect_equal(result$size(2), 5L)
  expect_equal(torch::as_array(result[, 5]), 0)
})

test_that("box_xyxy_to_xyxyr handles empty boxes", {
  empty <- torch::torch_zeros(c(0, 4))
  result <- box_xyxy_to_xyxyr(empty)

  expect_equal(result$size(1), 0L)
  expect_equal(result$size(2), 5L)
})

test_that("box_xyxy_to_xyxyr preserves dtype", {
  boxes <- torch::torch_tensor(matrix(c(1, 2, 3, 4), ncol = 4), dtype = torch::torch_int())
  result <- box_xyxy_to_xyxyr(boxes)

  expect_equal(result$dtype, torch::torch_int())
})

