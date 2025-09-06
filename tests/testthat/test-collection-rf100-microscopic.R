context("dataset-rf100-microscopic")

t <- withr::local_tempdir()

test_that("rf100_microscopic_collection handles missing files gracefully", {
  expect_error(
    rf100_microscopic_collection(dataset = "stomata_cell", split = "train", download = FALSE),
    class = "runtime_error"
  )
})

small_datasets <- c("blood_cell",  "cell", "liquid_crystals", "bacteria", "mitosis", "asbestos")

for (ds_name in small_datasets) {
  test_that(paste0("rf100_microscopic_collection loads ", ds_name, " correctly"), {
    ds <- rf100_microscopic_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_microscopic_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_gt(length(unique(ds$classes)), 1)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("labels", "boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

datasets <- c(
  "stomata_cell", "blood_cell", "parasite", "cell",
  "liquid_crystals", "bacteria", "cotton_desease",
  "mitosis", "phage", "liver_desease", "asbestos"
)

for (ds_name in datasets) {
  test_that(paste0("rf100_microscopic_collection loads ", ds_name, " correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
            "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_microscopic_collection(dataset = ds_name, split = "val", download = TRUE)

    expect_s3_class(ds, "rf100_microscopic_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_gt(length(unique(ds$classes)), 1)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("labels", "boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

