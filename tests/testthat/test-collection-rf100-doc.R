context("dataset-rf100-doc")

test_that("rf100_document_collection handles missing dataset gracefully", {
  expect_error(
    rf100_document_collection(dataset = "paragraph", split = "train", download = FALSE),
    "'arg' should be one of"
  )
})

datasets <- c(
  "tweeter_post", "tweeter_profile", "document_part",
  "activity_diagram", "signature"
)

for (ds_name in datasets) {
  test_that(paste0("rf100_document_collection loads ", ds_name, " correctly"), {
    ds <- rf100_document_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_document_collection")
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

test_that(paste0("rf100_document_collection loads paper_part correctly"), {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  ds <- rf100_document_collection(dataset = "paper_part", split = "train", download = TRUE)

  expect_s3_class(ds, "rf100_document_collection")
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


test_that("rf100_document_collection datasets can be turned into a dataloader", {
  ds <- rf100_document_collection(dataset = "document_part", split = "test", download = TRUE)

  expect_equal(ds$.length(), 318)
  expect_type(ds$classes, "character")
  expect_equal(length(unique(ds$classes)), 3)

  dl <- dataloader(ds, batch_size = 10, shuffle = TRUE)
  # 318k turns into 32 batches of 10
  expect_length(dl, 32)
  iter <- dataloader_make_iter(dl)
  expect_no_error(
    i <- dataloader_next(iter)
  )
  # Check shape, dtype, and values on X
  expect_equal(dim(i$x[[1]]), c(640, 640, 3))
  expect_tensor_dtype(i[[1]], torch_float())
  expect_true((torch_max(i[[1]]) <= 1)$item())
  # Check shape, dtype and names on y
  expect_length(i[[2]],10)
  expect_named(i, c("x", "y"))

})
