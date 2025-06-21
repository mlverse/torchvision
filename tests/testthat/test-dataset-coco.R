tmp <- tempfile()
dir.create(tmp)
withr::defer(unlink(tmp, recursive = TRUE), teardown_env())

collate_fn <- function(batch) {
  x <- lapply(batch, function(x) x$x)
  y <- lapply(batch, function(x) x$y)
  list(x = x, y = y)
}

test_that("coco_detection_dataset handles missing files gracefully", {
  expect_error(
    coco_detection_dataset(root = tmp, train = TRUE, year = "2017", download = FALSE),
    class = "runtime_error"
  )
})

test_that("coco_detection_dataset loads a single example correctly", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  expect_s3_class(ds, "coco_detection_dataset")
  expect_gt(length(ds), 0)

  example <- ds[1]
  x <- example$x
  y <- example$y

  expect_tensor(x)
  expect_equal(x$ndim, 3)

  expect_true(is.list(y))
  expect_setequal(names(y), c("boxes", "labels", "masks"))

  expect_tensor(y$boxes)
  expect_equal(y$boxes$ndim, 2)
  expect_equal(y$boxes$size(2), 4)

  expect_true(is.character(y$labels))
  expect_tensor(y$masks)
  expect_equal(y$masks$ndim, 3)
})

test_that("coco_detection_dataset batches correctly using dataloader", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  ds <- coco_detection_dataset(root = tmp, train = FALSE, year = "2017", download = TRUE)

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$x, "list")
  expect_true(all(vapply(batch$x, is_torch_tensor, logical(1))))

  expect_type(batch$y, "list")
  expect_setequal(names(batch$y[[1]]), c("boxes", "labels", "masks"))
  expect_tensor(batch$y[[1]]$boxes)
  expect_equal(batch$y[[1]]$boxes$ndim, 2)
  expect_equal(batch$y[[1]]$boxes$size(2), 4)
})

test_that("coco_caption_dataset handles missing files gracefully", {
  expect_error(
    coco_caption_dataset(root = tmp, train = TRUE, download = FALSE),
    class = "rlang_error"
  )
})

test_that("coco_caption_dataset loads a single example correctly", {
  skip_if(Sys.getenv("COCO_DATASET_TEST") != "1", "Set COCO_DATASET_TEST=1 to run")

  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  expect_s3_class(ds, "coco_caption_dataset")
  expect_gt(length(ds), 0)

  example <- ds[1]
  x <- example$x
  y <- example$y

  expect_true(is.array(x))
  expect_equal(length(dim(x)), 3)
  expect_equal(dim(x)[3], 3)

  expect_true(is.character(y))
  expect_gt(nchar(y), 0)
})

test_that("coco_caption_dataset batches correctly using dataloader", {
  skip_if(Sys.getenv("COCO_DATASET_TEST") != "1", "Set COCO_DATASET_TEST=1 to run")

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
