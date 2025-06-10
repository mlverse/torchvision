test_that("ms_coco_captions_dataset works", {
  skip_if_not_installed("magick")
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("withr")

  # Skip on CI unless explicitly requested (due to large download)
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""))

  dir <- tempfile()

  # Test dataset creation and download
  expect_message(
    dataset <- ms_coco_captions_dataset(root = dir, split = "val", download = TRUE),
    "Downloading"
  )

  expect_true(inherits(dataset, "ms_coco_captions_dataset"))
  expect_true(length(dataset) > 0)

  # Test sample retrieval
  sample <- dataset[1]
  expect_true("image" %in% names(sample))
  expect_true("caption" %in% names(sample))
  expect_true("image_id" %in% names(sample))

  # Test image properties
  expect_true(is.array(sample$image))
  expect_equal(length(dim(sample$image)), 3) # height, width, channels
  expect_equal(dim(sample$image)[3], 3) # RGB channels

  # Test caption properties
  expect_true(is.character(sample$caption))
  expect_true(nchar(sample$caption) > 0)

  # Test image_id properties
  expect_true(is.numeric(sample$image_id))

  # Test error handling for out-of-bounds access
  expect_error(dataset[0], "Index out of bounds")
  expect_error(dataset[length(dataset) + 1], "Index out of bounds")

  # Clean up
  unlink(dir, recursive = TRUE)
})

test_that("ms_coco_captions_dataset handles missing files gracefully", {
  skip_if_not_installed("jsonlite")

  dir <- tempfile()

  # Test error when files don't exist and download=FALSE
  expect_error(
    ms_coco_captions_dataset(root = dir, split = "val", download = FALSE),
    "Dataset files not found"
  )

  # Clean up
  unlink(dir, recursive = TRUE)
})

test_that("ms_coco_captions_dataset works with train split", {
  skip_if_not_installed("magick")
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("withr")

  # Skip on CI unless explicitly requested
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""))

  dir <- tempfile()

  dataset <- ms_coco_captions_dataset(root = dir, split = "train", download = TRUE)
  expect_true(inherits(dataset, "ms_coco_captions_dataset"))
  expect_true(length(dataset) > 0)

  sample <- dataset[1]
  expect_true(all(c("image", "caption", "image_id") %in% names(sample)))

  # Clean up
  unlink(dir, recursive = TRUE)
})
