context("dataset-rf100-biology")

t <- withr::local_tempdir()

test_that("rf100_biology_collection handles missing files gracefully", {
  expect_error(
    rf100_biology_collection(dataset = "stomata_cell", split = "train", download = FALSE),
    class = "runtime_error"
  )
})

small_dataset <- data.frame(name = c("blood_cell",  "cell", "bacteria", "mitosis"),
                            nlevels = c(3L, 1L, 1L, 1L)
)

for (ds_name in small_dataset$name) {
  test_that(paste0("rf100_biology_collection loads ", ds_name, " correctly"), {
    ds <- rf100_biology_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_biology_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_length(unique(ds$classes),
                  small_dataset[small_dataset$name == ds_name,]$nlevels)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("image_id","labels","boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

dataset <- data.frame(name = c("stomata_cell", "parasite", "cotton_desease", "phage", "liver_desease", "moth"),
                      nlevels = c(2L, 8L, 1L, 2L, 4L, 28L)
)

for (ds_name in dataset$name) {
  test_that(paste0("rf100_biology_collection loads ", ds_name, " correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
            "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_biology_collection(dataset = ds_name, split = "val", download = TRUE)

    expect_s3_class(ds, "rf100_biology_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_length(unique(ds$classes), dataset[dataset$name == ds_name,]$nlevels)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("image_id","labels","boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

