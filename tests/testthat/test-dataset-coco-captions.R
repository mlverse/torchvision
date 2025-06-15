test_that("ms_coco_captions_dataset works", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  dir <- tempfile()

  dataset <- ms_coco_captions_dataset(root = dir, train = FALSE, download = TRUE)

  expect_s3_class(dataset, "ms_coco_captions_dataset")
  expect_gt(length(dataset), 0)

  sample <- dataset[1]
  expect_true("image" %in% names(sample))
  expect_true("caption" %in% names(sample))
  expect_true("image_id" %in% names(sample))

  expect_true(is.array(sample$image))
  expect_equal(length(dim(sample$image)), 3)
  expect_equal(dim(sample$image)[3], 3)

  expect_true(is.character(sample$caption))
  expect_true(nchar(sample$caption) > 0)
  expect_true(is.numeric(sample$image_id))

  expect_error(dataset[0], "Index out of bounds")
  expect_error(dataset[length(dataset) + 1], "Index out of bounds")

  unlink(dir, recursive = TRUE)
})

test_that("ms_coco_captions_dataset handles missing files gracefully", {
  dir <- tempfile()

  expect_error(
    ms_coco_captions_dataset(root = dir, train = FALSE, download = FALSE),
    "Dataset files not found"
  )

  unlink(dir, recursive = TRUE)
})

test_that("ms_coco_captions_dataset works with train split", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  dir <- tempfile()

  dataset <- ms_coco_captions_dataset(root = dir, train = TRUE, download = TRUE)

  expect_s3_class(dataset, "ms_coco_captions_dataset")
  expect_gt(length(dataset), 0)

  sample <- dataset[1]
  expect_true(all(c("image", "caption", "image_id") %in% names(sample)))

  unlink(dir, recursive = TRUE)
})
