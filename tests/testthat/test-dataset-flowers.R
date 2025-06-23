context("dataset-flowers")

t <- withr::local_tempdir()

test_that("tests for the Flowers102 dataset for train split", {

  expect_error(
    flowers102_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  flowers <- flowers102_dataset(root = t, split = "train", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,1131000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for test split", {

  flowers <- flowers102_dataset(root = t, split = "test", download = TRUE)
  expect_length(flowers, 6149)
  first_item <- flowers[1]
  expect_length(first_item$x,784500)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for validation split", {

  flowers <- flowers102_dataset(root = t, split = "val", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,909000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Flowers102 dataset for dataloader", {

  resize_collate_fn <- function(batch) {
    target_size <- c(224, 224)
    xs <- lapply(batch, function(item) {
      transform_resize(item$x, target_size)
    })
    xs <- torch_stack(xs)
    ys <- lapply(batch, function(item) item$y)
    list(x = xs, y = ys)
  }
  dl <- dataloader(
    dataset = flowers102_dataset(root = t, transform = transform_to_tensor),
    batch_size = 4,
    collate_fn = resize_collate_fn
  )
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"list")
  expect_length(batch$y, 4)
  expect_equal(batch$y[[1]],1)
  expect_equal(batch$y[[2]],1)
  expect_equal(batch$y[[3]],1)
  expect_equal(batch$y[[4]],1)
})