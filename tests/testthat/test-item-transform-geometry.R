test_that("item_transform_rotate expands canvas for non-axis-aligned angles", {
  img <- torch_randn(3, 300, 500)
  after <- item_transform_rotate(img, angle = 30)

  new_W <- as.integer(ceiling(500 * abs(cos(30 * pi / 180)) + 300 * abs(sin(30 * pi / 180))))
  new_H <- as.integer(ceiling(500 * abs(sin(30 * pi / 180)) + 300 * abs(cos(30 * pi / 180))))

  expect_equal(after$shape[1], 3)
  expect_equal(after$shape[2], new_H)
  expect_equal(after$shape[3], new_W)
})

test_that("item_transform_rotate 0 degrees is identity", {
  img <- torch_randn(3, 200, 400)
  after <- item_transform_rotate(img, angle = 0)

  expect_equal(after$shape, c(3, 200, 400))
  expect_true(torch_allclose(after, img, atol = 0.01))
})

test_that("item_transform_rotate 90 degrees swaps dimensions", {
  img <- torch_randn(3, 200, 400)
  after <- item_transform_rotate(img, angle = 90)

  expect_equal(after$shape, c(3, 400, 200))
})

test_that("item_transform_rotate 180 degrees preserves dimensions", {
  img <- torch_randn(3, 200, 400)
  after <- item_transform_rotate(img, angle = 180)

  expect_equal(after$shape, c(3, 200, 400))
})

test_that("item_transform_rotate negative angles work", {
  img <- torch_randn(3, 100, 200)
  after <- item_transform_rotate(img, angle = -45)

  new_W <- as.integer(ceiling(200 * abs(cos(-45 * pi / 180)) + 100 * abs(sin(-45 * pi / 180))))
  new_H <- as.integer(ceiling(200 * abs(sin(-45 * pi / 180)) + 100 * abs(cos(-45 * pi / 180))))

  expect_equal(after$shape[1], 3)
  expect_equal(after$shape[2], new_H)
  expect_equal(after$shape[3], new_W)
})

test_that("item_transform_rotate square image stays square at 45 degrees", {
  img <- torch_randn(3, 100, 100)
  after <- item_transform_rotate(img, angle = 45)

  new_size <- as.integer(ceiling(100 * abs(cos(45 * pi / 180)) + 100 * abs(sin(45 * pi / 180))))
  expect_equal(after$shape, c(3, new_size, new_size))
})

test_that("item_transform_rotate empty regions are black", {
  img <- torch_zeros(3, 10, 10)
  img[, 3:8, 3:8] <- 1
  after <- item_transform_rotate(img, angle = 45)

  expect_true(after$min()$item() >= 0)
})

test_that("item_transform_rotate accepts torch_tensor directly", {
  img <- torch_randn(3, 100, 200)
  after <- item_transform_rotate(img, angle = 45)
  expect_true(after$ndim == 3)
  expect_equal(after$shape[1], 3)
})
