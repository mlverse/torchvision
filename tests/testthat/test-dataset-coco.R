test_that("coco_detection_dataset handles missing files gracefully", {
  tmp <- tempfile()

  expect_error(
    coco_detection_dataset(root = tmp, train = TRUE, year = "2017", download = FALSE),
    class = "runtime_error"
  )
})

test_that("coco_detection_dataset loads a single sample correctly", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(ds, "coco_detection_dataset")
  expect_gt(length(ds), 0)

  sample <- ds[1]
  expect_type(sample, "list")
  expect_named(sample, c("image", "target"))

  img <- sample$image
  target <- sample$target

  expect_tensor(img)
  expect_equal(img$ndim, 3)

  expect_true(is.list(target))
  expect_setequal(names(target), c("boxes", "labels", "area", "iscrowd", "segmentation"))

  if (!inherits(target$boxes, "torch_tensor")) {
    expect_true(is.matrix(target$boxes))
    expect_equal(ncol(target$boxes), 4)
    expect_equal(colnames(target$boxes), c("x1", "y1", "x2", "y2"))
  }

  expect_tensor(target$labels)
  expect_tensor_dtype(target$labels, torch::torch_int())

  expect_tensor(target$area)
  expect_tensor_dtype(target$area, torch::torch_float())

  expect_tensor(target$iscrowd)
  expect_tensor_dtype(target$iscrowd, torch::torch_int())

})

collate_fn <- function(batch) {
  images <- lapply(batch, function(x) x$image)
  targets <- lapply(batch, function(x) x$target)
  list(image = images, target = targets)
}

test_that("coco_detection_dataset batches correctly using dataloader", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  collate_fn <- function(batch) {
    images <- lapply(batch, function(x) x$image)
    targets <- lapply(batch, function(x) x$target)
    list(image = images, target = targets)
  }

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$image, "list")
  expect_true(all(vapply(batch$image, is_torch_tensor, logical(1))))

  expect_type(batch$target, "list")
  expect_setequal(names(batch$target[[1]]), c("boxes", "labels", "area", "iscrowd", "segmentation"))
  expect_tensor(batch$target[[1]]$boxes)
  expect_equal(dim(batch$target[[1]]$boxes)[2], 4)
})

