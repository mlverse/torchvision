context("Rcpp Box Operations")

test_that("box_area_cpp works correctly", {
  boxes <- matrix(c(
    0, 0, 10, 10,
    5, 5, 15, 20,
    0, 0, 100, 50
  ), ncol = 4, byrow = TRUE)
  
  areas <- box_area_cpp(boxes)
  expect_equal(areas, c(100, 150, 5000))
})

test_that("single box works", {
  boxes <- matrix(c(0, 0, 20, 30), ncol = 4)
  expect_equal(box_area_cpp(boxes), 600)
})

test_that("float coordinates work", {
  boxes <- matrix(c(0.5, 0.5, 10.5, 10.5), ncol = 4)
  expect_equal(box_area_cpp(boxes), 100)
})

test_that("invalid boxes return zero", {
  boxes <- matrix(c(10, 10, 5, 5), ncol = 4)
  expect_equal(box_area_cpp(boxes), 0)
})

test_that("wrong dimensions throw error", {
  boxes <- matrix(c(0, 0, 10), ncol = 3)
  expect_error(box_area_cpp(boxes))
})

test_that("wrapper works with matrix", {
  boxes <- matrix(c(0, 0, 10, 10, 5, 5, 15, 20), ncol = 4, byrow = TRUE)
  expect_equal(box_area_fast(boxes), c(100, 150))
})

test_that("wrapper works with data.frame", {
  boxes_df <- data.frame(x1 = c(0, 5), y1 = c(0, 5), 
                         x2 = c(10, 15), y2 = c(10, 20))
  expect_equal(box_area_fast(boxes_df), c(100, 150))
})

test_that("wrapper validates input", {
  expect_error(box_area_fast(c(0, 0, 10, 10)))
})

test_that("matches existing box_area function", {
  skip_if_not_installed("torch")
  
  boxes <- matrix(c(0, 0, 10, 10, 5, 5, 15, 20), ncol = 4, byrow = TRUE)
  boxes_tensor <- torch::torch_tensor(boxes, dtype = torch::torch_float())
  
  expect_equal(as.numeric(box_area(boxes_tensor)), box_area_fast(boxes))
})
