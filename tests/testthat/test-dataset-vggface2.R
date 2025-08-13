context('dataset-vggface2')

t <- withr::local_tempdir()
options(timeout = 60000)

test_that("VGGFace2 dataset works correctly for train split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  vgg <- vggface2_dataset(root = t, download = TRUE)
  expect_length(vgg, 3141890)
  first_item <- vgg[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})

test_that("VGGFace2 dataset works correctly for test split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  vgg <- vggface2_dataset(root = t, train = FALSE)
  expect_length(vgg, 169396)
  first_item <- vgg[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})