test_that("coco_caption_dataset handles missing files gracefully", {
  tmp <- tempfile()

  expect_error(
    coco_caption_dataset(root = tmp, train = TRUE, download = FALSE),
    class = "rlang_error"
  )

  unlink(tmp, recursive = TRUE)
})

test_that("coco_caption_dataset loads a single sample correctly", {
  skip_if(Sys.getenv("COCO_DATASET_TEST") != "1", "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  expect_true(inherits(ds, "coco_caption_dataset"))
  expect_gt(length(ds), 0)

  sample <- ds[1]
  expect_type(sample, "list")
  expect_named(sample, c("x", "y"))  # updated return keys

  img <- sample$x
  caption <- sample$y

  expect_true(is.array(img))
  expect_equal(length(dim(img)), 3)
  expect_equal(dim(img)[3], 3)

  expect_true(is.character(caption))
  expect_gt(nchar(caption), 0)
})

test_that("coco_caption_dataset batches correctly using dataloader", {
  skip_if(Sys.getenv("COCO_DATASET_TEST") != "1", "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  collate_fn <- function(batch) {
    x <- lapply(batch, function(item) item$x)
    y <- lapply(batch, function(item) item$y)
    list(x = x, y = y)
  }

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$x, "list")
  expect_true(all(vapply(batch$x, is.array, logical(1))))
  expect_true(all(vapply(batch$x, function(x) length(dim(x)) == 3, logical(1))))

  expect_type(batch$y, "list")
  expect_true(all(vapply(batch$y, is.character, logical(1))))
})
