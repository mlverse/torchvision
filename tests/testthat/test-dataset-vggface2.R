context('dataset-vggface2')

t <- withr::local_tempdir()
options(timeout = 60000)


test_that("VGGFace2 dataset works correctly for test split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  vgg <- vggface2_dataset(root = t, split = "test", download = TRUE)
  expect_length(vgg, 169396)
  expect_all_true(c("img_path", "labels", "class_to_idx", "identity_df") %in% names(vgg))
  expect_is(vgg$img_path, "data.frame")
  expect_is(vgg$identity_df, "data.frame")

  first_item <- vgg[1000]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 3)
  expect_equal(names(first_item$y), "n000029")
})
test_that("VGGFace2 dataset works correctly for train split", {

  skip_if(Sys.getenv("TEST_HUGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_HUGE_DATASETS=1 to enable tests requiring large downloads.")

  vgg <- vggface2_dataset(root = t, split = "train", download = TRUE)
  expect_length(vgg, 3141890)
  expect_all_true(c("img_path", "labels", "class_to_idx", "identity_df") %in% names(vgg))
  expect_is(vgg$img_path, "data.frame")
  expect_is(vgg$identity_df, "data.frame")

  first_item <- vgg[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)
})
