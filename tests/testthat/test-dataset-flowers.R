context("dataset-flowers")

t <- withr::local_tempdir()

test_that("tests for the Flowers102 dataset", {

  expect_error(
    flowers102_dataset(root = tempfile(), download = FALSE),
    class = "runtime_error"
  )

  flowers <- flowers102_dataset(root = t, split = "train", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,1131000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_equal(first_item$y, "pink primrose")

  flowers <- flowers102_dataset(root = t, split = "test", download = TRUE)
  expect_length(flowers, 6149)
  first_item <- flowers[1]
  expect_length(first_item$x,784500)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_equal(first_item$y, "pink primrose")

  flowers <- flowers102_dataset(root = t, split = "val", download = TRUE)
  expect_length(flowers, 1020)
  first_item <- flowers[1]
  expect_length(first_item$x,909000)
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_equal(first_item$y, "pink primrose")

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- sapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  dl <- torch::dataloader(
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
  expect_type(batch$y,"character")
  expect_length(batch$y, 4)
  expect_equal(batch$y[1],"pink primrose")
  expect_equal(batch$y[2],"pink primrose")
  expect_equal(batch$y[3],"pink primrose")
  expect_equal(batch$y[4],"pink primrose")

})