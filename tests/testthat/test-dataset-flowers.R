context("dataset-flowers")

test_that("tests for the Flowers102 dataset", {
  t <- tempfile()

  expect_error(
    flowers102_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  flowers <- flowers102_dataset(root = t, split = "train", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item[[2]], "pink primrose")

  flowers <- flowers102_dataset(root = t, split = "test", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item[[2]], "pink primrose")

  flowers <- flowers102_dataset(root = t, split = "val", download = TRUE)
  expect_gt(length(flowers), 0)
  first_item <- flowers[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item[[2]], "pink primrose")

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- sapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  dl <- torch::dataloader(
    dataset = flowers102_dataset(root = t, split = "train", download = TRUE),
    batch_size = 4,
    shuffle = TRUE,
    collate_fn = resize_collate_fn
  )
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_true(inherits(batch$x, "torch_tensor"))
  expect_equal(dim(batch$x)[1], 4)
  expect_equal(length(batch$y), 4)

})