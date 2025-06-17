test_that("coco_detection_dataset  handles missing files gracefully", {
  tmp <- tempfile()

  expect_error(
    coco_detection_dataset(root = tmp, train = TRUE, year = "2017", download = FALSE),
    class = "runtime_error"
  )
})

test_that("coco_detection_dataset loads correctly", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()

  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(ds, "coco_detection_dataset")
  expect_gt(length(ds), 0)

  el <- ds[1]

  expect_type(el, "list")
  expect_named(el, c("image", "target"))

  img <- el$image
  target <- el$target

  expect_true(inherits(img, "torch_tensor"))
  expect_equal(length(dim(img)), 3)

  expect_true(is.list(target))

  dl <- dataloader(ds, batch_size = 1)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_true(inherits(batch$image, "torch_tensor"))
  expect_equal(batch$image$ndim, 4) # [B, C, H, W]
  expect_equal(batch$image$dtype, torch::torch_float())

  expect_true(is.list(batch$target))

  expect_true(all(c("boxes", "labels", "area", "iscrowd", "segmentation") %in% names(target)))

  if (is.matrix(target$boxes) && nrow(target$boxes) > 0) {
    expect_equal(ncol(target$boxes), 4)
    expect_equal(colnames(target$boxes), c("x1", "y1", "x2", "y2"))
  }

  expect_true(is.numeric(target$labels))
  expect_true(is.numeric(target$area))
  expect_true(is.numeric(target$iscrowd))
})

test_that("coco_detection_dataset parameter validation", {
  tmp <- tempfile()

  expect_error(
    coco_detection_dataset(root = tmp, year = "2020"),
    "should be one of"
  )
})
