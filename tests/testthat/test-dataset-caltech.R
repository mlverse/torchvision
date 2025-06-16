context("dataset-caltech")

test_that("Caltech101 dataset works correctly", {
  t <- tempdir()

  expect_error(
    caltech101_dataset(root = tempfile(), download = FALSE),
    class = "runtime_error"
  )

  ds_category <- caltech101_dataset(root = t, target_type = "category", download = TRUE)
  expect_equal(length(ds_category), 8677)
  first_item <- ds_category[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_true(first_item$y %in% ds_category$classes)

  ds_annotation <- caltech101_dataset(root = t, target_type = "annotation")
  expect_equal(length(ds_annotation), 8677)
  first_item <- ds_annotation[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y$box_coord, "double")
  expect_length(first_item$y$box_coord,4)
  expect_type(first_item$y$obj_contour,"double")

  ds_all <- caltech101_dataset(root = t, target_type = "all")
  expect_equal(length(ds_all), 8677)
  first_item <- ds_all[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y$label, "character")
  expect_true(first_item$y$label %in% ds_all$classes)
  expect_type(first_item$y$box_coord, "double")
  expect_length(first_item$y$box_coord,4)
  expect_type(first_item$y$obj_contour,"double")

  resize_collate_fn <- function(batch) {
    target_size <- c(224, 224)
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, target_size)
    })
    xs <- torch::torch_stack(lapply(xs, torch::torch_tensor))
    ys <- lapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  ds <- caltech101_dataset(
    root = t,
    target_type = "category",
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
  expect_true(batch$y[[1]] %in% ds$classes)
  expect_true(batch$y[[2]] %in% ds$classes)
  expect_true(batch$y[[3]] %in% ds$classes)
  expect_true(batch$y[[4]] %in% ds$classes)
})

test_that("Caltech256 dataset works correctly", {
  t <- tempdir()

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