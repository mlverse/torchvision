context("dataset-coco")

tmp <- withr::local_tempdir()

collate_fn <- function(batch) {
  x <- lapply(batch, function(x) x$x)
  y <- lapply(batch, function(x) x$y)
  list(x = x, y = y)
}

test_that("coco_detection_dataset handles missing files gracefully", {
  expect_error(
    coco_detection_dataset(root = tempfile(), train = TRUE, year = "2017", download = FALSE),
    class = "runtime_error"
  )
})

test_that("coco_detection_dataset loads a single example correctly", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(ds, "coco_detection_dataset")
  expect_gt(length(ds), 0)

  item <- ds[1]
  y <- item$y

  expect_is(item$x, "array")
  expect_length(dim(item$x), 3)

  expect_type(y, "list")
  expect_named(y, c("boxes", "labels", "area", "iscrowd", "segmentation", "masks"))

  expect_tensor(y$boxes)
  expect_identical(y$boxes$ndim, 2)
  expect_identical(y$boxes$size(2), 4)

  expect_type(y$labels, "character")
  expect_gt(length(y$labels), 0)

  expect_tensor(y$area)
  expect_tensor(y$iscrowd)
  expect_true(is.list(y$segmentation))
  expect_tensor(y$masks)
  expect_identical(y$masks$ndim, 3)
})

test_that("coco_detection_dataset batches correctly using dataloader", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")


  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE, transform = transform_to_tensor)

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$x, "list")
  expect_true(all(vapply(batch$x, is_torch_tensor, logical(1))))

  expect_type(batch$y, "list")
  expect_named(batch$y[[1]], c("boxes", "labels", "area", "iscrowd", "segmentation", "masks"))
  expect_tensor(batch$y[[1]]$boxes)
  expect_identical(batch$y[[1]]$boxes$ndim, 2)
  expect_identical(batch$y[[1]]$boxes$size(2), 4)
})

test_that("coco_caption_dataset handles missing files gracefully", {
  expect_error(
    coco_caption_dataset(root = tempfile(), train = TRUE, download = FALSE),
    class = "rlang_error"
  )
})

test_that("coco_caption_dataset loads a single example correctly", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  expect_s3_class(ds, "coco_caption_dataset")
  expect_gt(length(ds), 0)

  item <- ds[1]
  x <- item$x
  y <- item$y

  expect_is(x, "array")
  expect_length(dim(x), 3)
  expect_identical(dim(x)[3], 3)

  expect_type(y, "character")
  expect_gt(nchar(y), 0)
})

test_that("coco_caption_dataset batches correctly using dataloader", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$x, "list")
  expect_true(all(vapply(batch$x, is.array, logical(1))))
  expect_true(all(vapply(batch$x, function(x) length(dim(x)) == 3, logical(1))))

  expect_type(batch$y, "list")
  expect_true(all(vapply(batch$y, is.character, logical(1))))
})
