context("dataset-lfw")

test_that("tests for the LFW-People dataset", {
  root_dir <- tempfile()
  
  expect_error(
    lfw_people <- lfw_people_dataset(dir)
  )

  lfw_people <- lfw_people_dataset( root = root_dir, split = "train", image_set = "deepfunneled", download = TRUE)
  expect_equal(length(lfw_people), 4857)
  first_item <- lfw_people[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal(first_item$y, "Aaron_Peirsol")

  lfw_people <- lfw_people_dataset( root = root_dir, split = "test", image_set = "deepfunneled", download = TRUE)
  expect_equal(length(lfw_people), 4857)
  first_item <- lfw_people[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal(first_item$y, "Aaron_Peirsol")

  lfw_people <- lfw_people_dataset( root = root_dir, split = "10fold", image_set = "deepfunneled", download = TRUE)
  expect_equal(length(lfw_people), 4857)
  first_item <- lfw_people[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "array"))
  expect_equal(first_item$y, "Aaron_Peirsol")

})