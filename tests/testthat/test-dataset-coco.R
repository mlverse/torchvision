test_that("coco_detection_dataset loads and returns expected fields", {
  skip_if_not_installed("magick")
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("withr")
  skip_if_not_installed("fs")

  # Skip on CI unless explicitly requested
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  dir <- tempfile()
  dataset <- coco_detection_dataset(root = dir, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(dataset, "coco_detection")  # More specific than inherits()
  expect_gt(length(dataset), 0)

  sample <- dataset[1]
  expect_type(sample, "list")
  expect_named(sample, c("image", "target"))  # More specific than %in%

  expect_true(is.array(sample$image))
  expect_equal(length(dim(sample$image)), 3)  # height, width, channels
  expect_true(is.list(sample$target))

  target <- sample$target
  expected_fields <- c("image_id", "boxes", "labels", "area", "iscrowd", "segmentation", "height", "width")
  expect_true(all(expected_fields %in% names(target)))

  # Your handling of empty boxes is perfect
  expect_true(is.matrix(target$boxes) || is.null(dim(target$boxes)))
  expect_true(is.numeric(target$labels))
  expect_true(is.numeric(target$area))
  expect_true(is.numeric(target$iscrowd))

  # Test bounding box format if present
  if (is.matrix(target$boxes) && nrow(target$boxes) > 0) {
    expect_equal(ncol(target$boxes), 4)
    expect_equal(colnames(target$boxes), c("x1", "y1", "x2", "y2"))
  }
})

test_that("coco_detection_dataset handles missing files gracefully", {
  skip_if_not_installed("fs")

  dir <- tempfile()

  # Should throw error if download=FALSE and data is not present
  expect_error(
    coco_detection_dataset(root = dir, train = TRUE, year = "2017", download = FALSE),
    "Dataset files not found"
  )
})

test_that("coco_detection_dataset parameter validation", {
  dir <- tempfile()

  # Test invalid year
  expect_error(
    coco_detection_dataset(root = dir, year = "2020"),
    "should be one of"
  )

  # Test valid years don't error on initialization (without download)
  expect_error(coco_detection_dataset(root = dir, year = "2017", download = FALSE), "Dataset files not found")
  expect_error(coco_detection_dataset(root = dir, year = "2016", download = FALSE), "Dataset files not found")
  expect_error(coco_detection_dataset(root = dir, year = "2014", download = FALSE), "Dataset files not found")
})
