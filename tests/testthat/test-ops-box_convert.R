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

test_that("box_xyxy_to_xyxyr with angle=0 preserves original coordinates", {
  result <- box_xyxy_to_xyxyr(xyxy, angle = 0)

  expect_tensor(result)
  expect_equal(result$size(2), 5L)
  expect_equal(result$size(1), 5L)
  expect_equal(torch::as_array(result[, 1:4]), torch::as_array(xyxy))
  expect_equal(torch::as_array(result[, 5]), rep(0, 5))
})

test_that("box_xyxy_to_xyxyr with angle=0 handles single box", {
  single <- torch::torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- box_xyxy_to_xyxyr(single, angle = 0)

  expect_equal(result$size(1), 1L)
  expect_equal(result$size(2), 5L)
  expect_equal(torch::as_array(result[, 5]), 0)
})

test_that("box_xyxy_to_xyxyr handles empty boxes", {
  empty <- torch::torch_zeros(c(0, 4))
  result <- box_xyxy_to_xyxyr(empty, angle = 0)

  expect_equal(result$size(1), 0L)
  expect_equal(result$size(2), 5L)
})

test_that("box_xyxy_to_xyxyr preserves dtype", {
  boxes <- torch::torch_tensor(matrix(c(1, 2, 3, 4), ncol = 4), dtype = torch::torch_int())
  result <- box_xyxy_to_xyxyr(boxes, angle = 0)

  expect_equal(result$dtype, torch::torch_int())
})

test_that("box_xyxy_to_xyxyr rotates box by angle around center", {
  # A 2x2 square centered at (0,0): xyxy = (-1, -1, 1, 1)
  # Rotating by pi/2 should give the same axis-aligned box
  box <- torch::torch_tensor(matrix(c(-1, -1, 1, 1), ncol = 4))
  result <- box_xyxy_to_xyxyr(box, angle = pi / 2)

  expect_equal(torch::as_array(result[, 1:4]), torch::as_array(box), tolerance = 1e-5)
  expect_equal(as.numeric(result[1, 5]$cpu()), pi / 2, tolerance = 1e-5)
})

test_that("box_xyxy_to_xyxyr rotates by pi/4 and produces larger enclosing box", {
  # A 2x2 square centered at origin: xyxy = (-1, -1, 1, 1)
  # Rotated by pi/4, the enclosing box expands to [-sqrt(2), -sqrt(2), sqrt(2), sqrt(2)]
  box <- torch::torch_tensor(matrix(c(-1, -1, 1, 1), ncol = 4))
  result <- box_xyxy_to_xyxyr(box, angle = pi / 4)

  expect_equal(as.numeric(result[1, 1]$cpu()), -sqrt(2), tolerance = 1e-5)
  expect_equal(as.numeric(result[1, 2]$cpu()), -sqrt(2), tolerance = 1e-5)
  expect_equal(as.numeric(result[1, 3]$cpu()), sqrt(2), tolerance = 1e-5)
  expect_equal(as.numeric(result[1, 4]$cpu()), sqrt(2), tolerance = 1e-5)
  expect_equal(as.numeric(result[1, 5]$cpu()), pi / 4, tolerance = 1e-5)
})

test_that("box_xyxy_to_xyxyr accepts per-box angles", {
  box <- torch::torch_tensor(rbind(c(0, 0, 2, 2), c(0, 0, 2, 2)))
  angles <- torch::torch_tensor(c(0, pi / 4))
  result <- box_xyxy_to_xyxyr(box, angle = angles)

  expect_equal(result$size(1), 2L)
  expect_equal(as.numeric(result[1, 5]$cpu()), 0, tolerance = 1e-5)
  expect_equal(as.numeric(result[2, 5]$cpu()), pi / 4, tolerance = 1e-5)
})

