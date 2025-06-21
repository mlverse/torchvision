context("dataset-caltech")

t <- withr::local_tempdir()

test_that("Caltech101 dataset works correctly", {

  expect_error(
    caltech101_dataset(root = tempfile(), download = FALSE),
    class = "runtime_error"
  )

  ds <- caltech101_dataset(root = t, download = TRUE)
  expect_equal(length(ds), 8677)
  first_item <- ds[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_length(first_item$x, 234000)
  expect_type(first_item$y,"list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_float())
  expect_type(first_item$y$labels, "integer")
  expect_equal(first_item$y$labels,1)
  expect_tensor(first_item$y$contour)
  expect_tensor_shape(first_item$y$contour,c(1,20,2))
  expect_tensor_dtype(first_item$y$contour,torch_float())
})

test_that("Caltech101 dataset works correctly (dataloader)", {

  resize_collate_fn <- function(batch) {
    target_size <- c(224, 224)
    xs <- lapply(batch, function(sample) {
      img <- sample$x
      img <- torch_tensor(img)
      transform_resize(img, target_size)
    })
    xs <- torch::torch_stack(xs)
    ys <- lapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  ds <- caltech101_dataset(root = t)
  dl <- dataloader(ds, batch_size = 4, collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"list")
  expect_type(batch$y[[1]],"list")
  expect_tensor(batch$y[[1]]$boxes)
  expect_tensor_shape(batch$y[[1]]$boxes,c(1,4))
  expect_tensor_dtype(batch$y[[1]]$boxes,torch_float())
  expect_equal(batch$y[[1]]$labels,1)
  expect_tensor(batch$y[[1]]$contour)
  expect_tensor_shape(batch$y[[1]]$contour,c(1,20,2))
  expect_tensor_dtype(batch$y[[1]]$contour,torch_float())
})

test_that("Caltech256 dataset works correctly", {

  expect_error(
    caltech256_dataset(root = tempfile(), download = FALSE),
    class = "runtime_error"
  )

  caltech256 <- caltech256_dataset(root = t, download = TRUE)
  expect_length(caltech256, 30607)
  first_item <- caltech256[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,416166)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_equal(first_item$y, "ak47")

  resize_collate_fn <- function(batch) {
    target_size <- c(224, 224)
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, target_size)
    })
    xs <- torch::torch_stack(lapply(xs, torch::torch_tensor))
    ys <- lapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  ds <- caltech256_dataset(
    root = t,
    transform = transform_to_tensor
  )
  dl <- dataloader(ds, batch_size = 4, collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"list")
  expect_type(batch$y[[1]],"character")
  expect_equal(batch$y[[1]],"ak47")
  expect_equal(batch$y[[2]],"ak47")
  expect_equal(batch$y[[3]],"ak47")
  expect_equal(batch$y[[4]],"ak47")
  
})