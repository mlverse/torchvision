test_that("places365_dataset loads and returns valid items", {

  skip_on_cran()
  skip_on_os("windows")

  ds <- places365_dataset(
    split = "val",
    download = TRUE,
    transform = transform_to_tensor
  )

  expect_s3_class(ds, "places365_dataset")
  expect_length(ds, 36500)

  item <- ds[1]

  expect_type(item, "list")
  expect_named(item, c("x", "y"))

  # Check image
  expect_s3_class(item$x, "torch_tensor")
  expect_equal(item$x$dim(), 3)
  expect_equal(item$x$size(1), 3)  # Channel
  expect_true(item$x$size(2) > 0)  # Height
  expect_true(item$x$size(3) > 0)  # Width
  expect_gte(torch::torch_min(item$x)$item(), 0)
  expect_lte(torch::torch_max(item$x)$item(), 1)

  # Check label
  expect_true(is.numeric(item$y))
  expect_gte(item$y, 1)
  expect_lte(item$y, 365)
})

test_that("places365 categories are correctly mapped", {
  ann_path <- file.path(tempdir(), "places365", "categories_places365.txt")
  skip_if_not(file.exists(ann_path), message = "Category file missing")

  categories <- suppressWarnings(readLines(ann_path))
  expect_length(categories, 365)

  ds <- places365_dataset(split = "val", download = FALSE)
  label <- ds[1]$y
  expect_true(label <= length(categories))
})
