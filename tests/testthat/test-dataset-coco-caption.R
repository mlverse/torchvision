test_that("coco_caption_dataset handles missing files gracefully", {
  tmp <- tempfile()

  expect_error(
    coco_caption_dataset(root = tmp, train = TRUE, download = FALSE),
    class = "rlang_error"
  )

  unlink(tmp, recursive = TRUE)
})

test_that("coco_caption_dataset loads a single sample correctly", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  expect_true(inherits(ds, "coco_caption_dataset"))
  expect_gt(length(ds), 0)

  sample <- ds[1]
  expect_type(sample, "list")
  expect_named(sample, c("image", "caption", "image_id"))

  img <- sample$image
  caption <- sample$caption
  image_id <- sample$image_id

  expect_true(is.array(img))
  expect_equal(length(dim(img)), 3)
  expect_equal(dim(img)[3], 3)

  expect_true(is.character(caption))
  expect_gt(nchar(caption), 0)

  expect_true(is.numeric(image_id))
})

test_that("coco_caption_dataset batches correctly using dataloader", {
  skip_if(identical(Sys.getenv("COCO_DATASET_TEST"), ""), "Set COCO_DATASET_TEST=1 to run")

  tmp <- tempfile()
  ds <- coco_caption_dataset(root = tmp, train = FALSE, download = TRUE)

  collate_fn <- function(batch) {
    images <- lapply(batch, function(x) x$image)
    captions <- lapply(batch, function(x) x$caption)
    image_ids <- lapply(batch, function(x) x$image_id)
    list(image = images, caption = captions, image_id = image_ids)
  }

  dl <- dataloader(ds, batch_size = 2, collate_fn = collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)

  expect_type(batch$image, "list")
  expect_true(all(vapply(batch$image, is.array, logical(1))))
  expect_true(all(vapply(batch$image, function(x) length(dim(x)) == 3, logical(1))))

  expect_type(batch$caption, "list")
  expect_true(all(vapply(batch$caption, is.character, logical(1))))

  expect_type(batch$image_id, "list")
  expect_true(all(vapply(batch$image_id, is.numeric, logical(1))))
})
