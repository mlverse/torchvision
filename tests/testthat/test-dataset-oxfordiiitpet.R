context("dataset-oxfordiiitpet")

test_that("tests for the Oxford-IIIT Pet dataset", {
  t <- tempfile()

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "category", train = TRUE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)
  expect_equal(first_item$class_name, "Abyssinian")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "binary-category", train = TRUE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)
  expect_equal(first_item$class_name, "Cat")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "segmentation", train = TRUE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "category", train = FALSE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)
  expect_equal(first_item$class_name, "Abyssinian")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "binary-category", train = FALSE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)
  expect_equal(first_item$class_name, "Cat")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "segmentation", train = FALSE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y","class_name"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, 1)

})