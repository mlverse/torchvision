context("dataset-fgcv")

test_that("tests for the FGCV-Aircraft dataset", {
  t <- tempfile()

  expect_error(
    fgcv_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = FALSE),
    class = "runtime_error"
  )

  fgcv <- fgcv_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "train", annotation_level = "family", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "train", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "test", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "test", annotation_level = "family", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "test", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "trainval", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "trainval", annotation_level = "family", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "trainval", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "val", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "val", annotation_level = "family", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)

  fgcv <- fgcv_aircraft_dataset(root = t, split = "val", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgcv), 0)
  first_item <- fgcv[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)
  
})
