test_that("dataset_coco_detection$new handles missing files gracefully", {
  tmp <- tempfile()

  expect_error(
    dataset_coco_detection$new(root = tmp, train = TRUE, year = "2017", download = FALSE),
    class = "rlang_error"
  )
})

test_that("dataset_coco_detection$new loads correctly", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()

  ds <- dataset_coco_detection$new(root = tmp, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(ds, "dataset_coco_detection$new")
  expect_gt(length(ds), 0)

  el <- ds[1]

  expect_type(el, "list")
  expect_named(el, c("image", "target"))

  img <- el$image
  target <- el$target

  expect_true(is.array(img))
  expect_equal(length(dim(img)), 3)

  expect_true(is.list(target))
  expect_true(all(c("image_id", "boxes", "labels", "area", "iscrowd", "segmentation", "height", "width") %in% names(target)))

  if (is.matrix(target$boxes) && nrow(target$boxes) > 0) {
    expect_equal(ncol(target$boxes), 4)
    expect_equal(colnames(target$boxes), c("x1", "y1", "x2", "y2"))
  }

  expect_true(is.numeric(target$labels))
  expect_true(is.numeric(target$area))
  expect_true(is.numeric(target$iscrowd))
})

test_that("dataset_coco_detection$new parameter validation", {
  tmp <- tempfile()

  expect_error(
    dataset_coco_detection$new(root = tmp, year = "2020"),
    "should be one of"
  )

  expect_error(
    dataset_coco_detection$new(root = tmp, year = "2017", download = FALSE),
    class = "rlang_error"
  )
})
