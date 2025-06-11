context("dataset-caltech")

test_that("Caltech101 dataset works correctly", {
  t <- tempfile()

  expect_error(
    caltech101_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  ds_category <- caltech101_dataset(root = t, target_type = "category", download = TRUE)
  expect_equal(length(ds_category), 8677)
  first_item <- ds_category[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_type(first_item$y,"character")

  ds_annotation <- caltech101_dataset(root = t, target_type = "annotation", download = TRUE)
  expect_equal(length(ds_annotation), 8677)
  first_item_ann <- ds_annotation[1]
  expect_named(first_item_ann, c("x", "y"))
  expect_true(inherits(first_item_ann$x, "torch_tensor"))
  expect_type(first_item_ann$y$box_coord, "double")
  expect_type(first_item_ann$y$obj_contour,"double" )

  dl <- dataloader(ds_category, batch_size = 4, collate_fn = function(batch) batch)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_equal(length(batch), 4)
  expect_true(all(vapply(batch, function(item) inherits(item$x, "torch_tensor"), logical(1))))
})

test_that("Caltech256 dataset works correctly", {
  t <- tempfile()

  expect_error(
    caltech256_dataset(root = t, download = FALSE),
    class = "runtime_error"
  )

  caltech256 <- caltech256_dataset(root = t, download = TRUE)
  expect_equal(length(caltech256), 30607)
  first_item <- caltech256[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_equal(first_item$y, "001.ak47")

  dl <- dataloader(caltech256, batch_size = 1)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_true(inherits(batch$x, "torch_tensor"))
  expect_equal(batch$x$shape[1], 1)
  expect_equal(batch$y, "001.ak47")
})