context('dataset-vggface2')

t <- withr::local_tempdir()

test_that("VGGFace2 dataset works correctly for train split", {

  vgg <- vggface2_dataset(root = t, download = TRUE)
  expect_length(vgg, 3141890)
  first_item <- vgg[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("VGGFace2 dataset works correctly for test split", {

  vgg <- vggface2_dataset(root = t, train = FALSE)
  expect_length(vgg, 169396)
  first_item <- vgg[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})