context("vector-and-matrix-addition")

test_that("add_vectors works correctly", {

  result3 <- add_vectors(c(1, 2, 3), c(4, 5, 6))

  expect_equal(result3, c(5, 7, 9))

})

test_that("add_matrices works correctly", {

  mat1 <- matrix(1:4, nrow = 2)
  mat2 <- matrix(5:8, nrow = 2)

  result4 <- add_matrices(mat1, mat2)

  expect_equal(result4, matrix(c(6, 8, 10, 12), nrow = 2))

})