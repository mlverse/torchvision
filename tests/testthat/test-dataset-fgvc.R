context("dataset-fgvc")

test_that("tests for the FGVC-Aircraft dataset", {
  t <- tempfile()

  expect_error(
    fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = FALSE),
    class = "runtime_error"
  )

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)
  expect_equal((first_item[[3]]), "707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "family", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)
  expect_equal((first_item[[3]]), "Boeing 707")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)
  expect_equal((first_item[[3]]), "Boeing")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)
  expect_equal((first_item[[3]]), "707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "family", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)
  expect_equal((first_item[[3]]), "Boeing 707")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)
  expect_equal((first_item[[3]]), "Boeing")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)
  expect_equal((first_item[[3]]), "707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "family", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)
  expect_equal((first_item[[3]]), "Boeing 707")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)
  expect_equal((first_item[[3]]), "Boeing")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "variant", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 1)
  expect_equal((first_item[[3]]), "707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "family", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 13)
  expect_equal((first_item[[3]]), "Boeing 707")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "manufacturer", download = TRUE)
  expect_gt(length(fgvc), 0)
  first_item <- fgvc[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item[[2]]), 5)
  expect_equal((first_item[[3]]), "Boeing")
  
})
