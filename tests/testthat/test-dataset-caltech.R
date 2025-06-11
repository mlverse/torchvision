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
  expect_equal(first_item$y,"accordion")

  ds_annotation <- caltech101_dataset(root = t, target_type = "annotation", download = TRUE)
  expect_equal(length(ds_annotation), 8677)
  first_item_ann <- ds_annotation[1]
  expect_named(first_item_ann, c("x", "y"))
  expect_true(inherits(first_item_ann$x, "torch_tensor"))
  expect_equal(first_item_ann$y$box_coord, c(2, 300, 1, 260))
  obj_contour <- matrix(c(
    37.1657459,  58.5930018,
    61.9447514,  44.2762431,
    89.4769797,  23.9023941,
    126.9208103,   0.7753223,
    169.3204420,   2.9779006,
    226.0368324,  61.3462247,
    259.0755064, 126.8729282,
    258.5248619, 214.9760589,
    203.4604052, 267.8379374,
    177.5801105, 270.5911602,
    147.8453039, 298.6740331,
    117.0092081, 298.6740331,
    1.3738490, 187.9944751,
    1.3738490,  94.9355433,
    7.9815838,  90.5303867,
    0.8232044,  77.3149171,
    16.2412523,  62.4475138,
    31.6593002,  62.9981584,
    38.8176796,  56.9410681,
    38.8176796,  56.9410681
  ), ncol = 2, byrow = TRUE)
  expect_equal(first_item_ann$y$obj_contour,obj_contour )

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