context("dataset-flowers")

test_that("tests for the Flowers102 dataset", {
  t <- tempfile()

  expect_error(
    flowers102_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  flowers <- flowers102_dataset(root = t, split = "train", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), "pink primrose")
  
  flowers <- flowers102_dataset(root = t, split = "test", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), "pink primrose")

  flowers <- flowers102_dataset(root = t, split = "val", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), "pink primrose")
  
})