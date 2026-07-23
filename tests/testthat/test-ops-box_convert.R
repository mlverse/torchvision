x1 <- torch_ones(5)
y1 <- torch_ones(5)
x2 <- x1 + 1
y2 <- y1 + 1
xyxy <- torch_stack(list(x1,y1,x2,y2))$transpose(2,1)

test_that("box_cxcywh_to_xyxy box_xyxy_to_cxcywh box_xywh_to_xyxy box_xyxy_to_xywh", {
  # from xyxy
  cxcywh <- box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(xyxy))
  xywh <- box_xywh_to_xyxy(box_xyxy_to_xywh(xyxy))

  expect_tensor(cxcywh)
  expect_tensor(xywh)
  expect_equal(as_array(cxcywh), as_array(xyxy))
  expect_equal(as_array(xywh), as_array(xyxy))
})

test_that("box_xyxy_to_xyxyr with angle=0 preserves original coordinates", {
  result <- box_xyxy_to_xyxyr(xyxy, angle = 0)

  expect_tensor(result)
  expect_tensor_shape(result, c(5, 5))
  expect_tensor_dtype(result, torch_float())
  expect_equal_to_r(result[, 1:4], as_array(xyxy))
  expect_equal_to_r(result[, 5], rep(0, 5))
})

test_that("box_xyxy_to_xyxyr with angle=0 handles single box", {
  single <- torch_tensor(matrix(c(10, 20, 50, 60), ncol = 4))
  result <- box_xyxy_to_xyxyr(single, angle = 0)

  expect_tensor_shape(result, c(1, 5))
  expect_tensor_dtype(result, torch_float())
  expect_equal_to_r(result[, 5], 0)
})

test_that("box_xyxy_to_xyxyr handles empty boxes", {
  empty <- torch_zeros(c(0, 4))
  result <- box_xyxy_to_xyxyr(empty, angle = 0)

  expect_tensor_shape(result, c(0, 5))
  expect_tensor_dtype(result, torch_float())
})

test_that("box_xyxy_to_xyxyr preserves dtype", {
  boxes <- torch_tensor(matrix(c(1, 2, 3, 4), ncol = 4), dtype = torch_float())
  result <- box_xyxy_to_xyxyr(boxes, angle = 0)

  expect_tensor_dtype(result, torch_float())
})

test_that("box_xyxy_to_xyxyr rotates box by 90 degrees around center", {
  # A 2x2 square centered at (0,0): xyxy = (-1, -1, 1, 1)
  # Rotating by 90 degrees should give the same axis-aligned box
  box <- torch_tensor(matrix(c(-1, -1, 1, 1), ncol = 4))
  result <- box_xyxy_to_xyxyr(box, angle = 90)

  expect_equal_to_r(result[, 1:4], as_array(box), tolerance = 1e-5)
  expect_equal_to_r(result[1, 5], 90, tolerance = 1e-5)
})

test_that("box_xyxy_to_xyxyr rotates by 45 degrees and produces larger enclosing box", {
  # A 2x2 square centered at origin: xyxy = (-1, -1, 1, 1)
  # Rotated by 45 degrees, the enclosing box expands to [-sqrt(2), -sqrt(2), sqrt(2), sqrt(2)]
  box <- torch_tensor(matrix(c(-1, -1, 1, 1), ncol = 4))
  result <- box_xyxy_to_xyxyr(box, angle = 45)

  expect_equal_to_r(result[1, 1], -sqrt(2), tolerance = 1e-5)
  expect_equal_to_r(result[1, 2], -sqrt(2), tolerance = 1e-5)
  expect_equal_to_r(result[1, 3], sqrt(2), tolerance = 1e-5)
  expect_equal_to_r(result[1, 4], sqrt(2), tolerance = 1e-5)
  expect_equal_to_r(result[1, 5], 45, tolerance = 1e-5)
})

test_that("box_xyxy_to_xyxyr accepts per-box angles", {
  box <- torch_tensor(rbind(c(0, 0, 2, 2), c(0, 0, 2, 2)))
  angles <- torch_tensor(c(0, 45))
  result <- box_xyxy_to_xyxyr(box, angle = angles)

  expect_tensor_shape(result, c(2, 5))
  expect_tensor_dtype(result, torch_float())
  expect_equal_to_r(result[1, 5], 0, tolerance = 1e-5)
  expect_equal_to_r(result[2, 5], 45, tolerance = 1e-5)
})
