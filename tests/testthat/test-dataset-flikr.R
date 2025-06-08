context("dataset-flikr")

test_that("tests for the Flikr8k dataset", {
  t <- tempfile()

  expect_error(
    flickr8k_dataset(root = t, download = FALSE)
  )

  flikr8k <- flickr8k_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(flikr8k), 6000)
  first_item <- flikr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "A black dog is running after a white dog in the snow .")
  expect_equal((first_item$y[[2]]), "Black dog chasing brown dog through snow")
  expect_equal((first_item$y[[3]]), "Two dogs chase each other across the snowy ground .")
  expect_equal((first_item$y[[4]]), "Two dogs play together in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs running through a low lying body of water .")

  flikr8k <- flickr8k_dataset(root = t, train = FALSE, download = TRUE)
  expect_equal(length(flikr8k), 1000)
  first_item <- flikr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "The dogs are in the snow in front of a fence .")
  expect_equal((first_item$y[[2]]), "The dogs play on the snow .")
  expect_equal((first_item$y[[3]]), "Two brown dogs playfully fight in the snow .")
  expect_equal((first_item$y[[4]]), "Two brown dogs wrestle in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs playing in the snow .")

})
