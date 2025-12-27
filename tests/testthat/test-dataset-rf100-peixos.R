context("dataset-rf100-peixos")

t <- withr::local_tempdir()

test_that("rf100_peixos_segmentation_dataset handles missing files gracefully", {
  expect_error(
    rf100_peixos_segmentation_dataset(split = "train", root = t, download = FALSE),
    class = "runtime_error"
  )
})

test_that("rf100_peixos_segmentation_dataset 'test' split works", {
  # windows error on tar deep folder decompression on tempdir()
  skip_on_os("windows")
  expect_no_error(
    ds <- rf100_peixos_segmentation_dataset(split = "test", root = t, download = TRUE)
  )
  expect_length(ds, 118)
  item <- ds[1]
  expect_tensor(item$y$masks)
  expect_identical(item$y$masks$ndim, 3)
  expect_gt(item$y$masks$sum()$item(), 0)
})

test_that("rf100_peixos_segmentation_dataset 'val' split works", {
  # windows error on tar deep folder decompression on tempdir()
  skip_on_os("windows")
  expect_no_error(
    ds <- rf100_peixos_segmentation_dataset(split = "val", root = t, download = TRUE)
  )
  expect_length(ds, 251)
  item <- ds[1]
  expect_tensor(item$y$masks)
  expect_identical(item$y$masks$ndim, 3)
  expect_gt(item$y$masks$sum()$item(), 0)
})


test_that(paste0("rf100_peixos_segmentation_dataset loads 'train' split correctly"), {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  # windows error on tar deep folder decompression on tempdir()
  skip_on_os("windows")
  ds <- rf100_peixos_segmentation_dataset(split = "train", root = t, download = TRUE)

  expect_s3_class(ds, "rf100_peixos_segmentation_dataset")
  expect_length(ds, 821)

  item <- ds[1]

  expect_type(item$y, "list")
  expect_named(item$y, c("masks", "labels"))
  expect_tensor(item$y$masks)
  expect_identical(item$y$masks$ndim, 3)
  expect_gt(item$y$masks$sum()$item(), 0)
  expect_identical(item$y$labels, 1L)
  expect_s3_class(item, "image_with_segmentation_mask")
})
