context("dataset-flickr")

test_that("tests for the flickr8k dataset", {
  t <- tempfile()

  flickr8k <- flickr8k_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(flickr8k), 6000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "A black dog is running after a white dog in the snow .")
  expect_equal((first_item$y[[2]]), "Black dog chasing brown dog through snow")
  expect_equal((first_item$y[[3]]), "Two dogs chase each other across the snowy ground .")
  expect_equal((first_item$y[[4]]), "Two dogs play together in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs running through a low lying body of water .")

  flickr8k <- flickr8k_dataset(root = t, train = FALSE, download = TRUE)
  expect_equal(length(flickr8k), 1000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "The dogs are in the snow in front of a fence .")
  expect_equal((first_item$y[[2]]), "The dogs play on the snow .")
  expect_equal((first_item$y[[3]]), "Two brown dogs playfully fight in the snow .")
  expect_equal((first_item$y[[4]]), "Two brown dogs wrestle in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs playing in the snow .")

})

test_that("tests for the flickr30k dataset", {
  t <- tempfile()

  flickr30k <- flickr30k_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(flickr30k), 29000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "Two young guys with shaggy hair look at their hands while hanging out in the yard.")
  expect_equal((first_item$y[[2]]), "Two young, White males are outside near many bushes.")
  expect_equal((first_item$y[[3]]), "Two men in green shirts are standing in a yard.")
  expect_equal((first_item$y[[4]]), "A man in a blue shirt standing in a garden.")
  expect_equal((first_item$y[[5]]), "Two friends enjoy time spent together.")

  flickr30k <- flickr30k_dataset(root = t, train = FALSE, download = TRUE)
  expect_equal(length(flickr30k), 1000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal((first_item$y[[1]]), "The man with pierced ears is wearing glasses and an orange hat.")
  expect_equal((first_item$y[[2]]), "A man with glasses is wearing a beer can crocheted hat.")
  expect_equal((first_item$y[[3]]), "A man with gauges and glasses is wearing a Blitz hat.")
  expect_equal((first_item$y[[4]]), "A man in an orange hat starring at something.")
  expect_equal((first_item$y[[5]]), "A man wears an orange hat and glasses.")

})