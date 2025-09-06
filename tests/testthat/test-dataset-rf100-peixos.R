context("dataset-rf100-peixos")

t <- withr::local_tempdir()

test_that("rf100_peixos_segmentation_dataset handles missing files gracefully", {
  expect_error(
    rf100_peixos_segmentation_dataset(split = "train", root = tempfile(), download = FALSE),
    class = "runtime_error"
  )
})

test_that("rf100_peixos_segmentation_dataset finds deeply nested directories", {
  root <- withr::local_tempdir()
  nested <- fs::path(root, "rf100-peixos/home/zuppif/peixos/peixos-fish/train")
  fs::dir_create(nested, recurse = TRUE)
  png::writePNG(array(0, dim = c(2, 2, 3)), fs::path(nested, "sample.png"))
  ann <- list(
    images = list(list(id = 1, file_name = "sample.png", width = 2, height = 2)),
    annotations = list(list(id = 1, image_id = 1, category_id = 1,
                            segmentation = list(list(c(0, 0, 0, 2, 2, 2, 2, 0))))),
    categories = list(list(id = 1, name = "fish"))
  )
  jsonlite::write_json(ann, fs::path(nested, "_annotations.coco.json"), auto_unbox = TRUE)

  ds <- rf100_peixos_segmentation_dataset(split = "train", root = root, download = FALSE)
  expect_length(ds, 1)
  item <- ds[1]
  expect_tensor(item$y$masks)
  expect_gt(item$y$masks$sum()$item(), 0)
})

splits <- c("train")

for (sp in splits) {
  test_that(paste0("rf100_peixos_segmentation_dataset loads ", sp, " split correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
            "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_peixos_segmentation_dataset(split = sp, root = t, download = TRUE)

    expect_s3_class(ds, "rf100_peixos_segmentation_dataset")
    expect_gt(length(ds), 0)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("masks", "labels"))
    expect_tensor(item$y$masks)
    expect_equal(item$y$masks$ndim, 3)
    expect_gt(item$y$masks$sum()$item(), 0)
    expect_equal(item$y$labels, 1L)
    expect_s3_class(item, "image_with_segmentation_mask")
  })
}
