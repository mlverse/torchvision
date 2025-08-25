context("dataset-rf100-doc")

t <- withr::local_tempdir()

test_that("rf100_document_collection handles missing files gracefully", {
  expect_error(
    rf100_document_collection(dataset = "tweeter_post", split = "train", root = tempfile(), download = FALSE),
    class = "runtime_error"
  )
})

datasets <- c(
  "tweeter_post", "tweeter_profile", "document_part",
  "activity_diagram", "signature", "paper_part",
  "tabular_data" #, "paragraph"
)

for (ds_name in datasets) {
  test_that(paste0("rf100_document_collection loads ", ds_name, " correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
            "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_document_collection(dataset = ds_name, split = "train", root = t, download = TRUE)

    expect_s3_class(ds, "rf100_document_collection")
    expect_gt(length(ds), 0)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("labels", "boxes"))
    expect_type(item$y$labels, "character")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

